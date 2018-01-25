optim = require 'optim'
require 'adam_gan'

-- load models and training criterions
local vae_encoder, vae_decoder, gan, sampling_z = table.unpack(models)
local KLD, ReconCriterion, BCECriterion = table.unpack(criterions)
local sampling_z2 = sampling_z:clone()
local optimStateVae_encoder, optimStateVae_decoder, optimStateGan

-- reload if previous checkpoint exists
-- otherwise initialize optimStates 
if opt.retrain ~= 'none' and opt.optimState ~= 'none' then
  local models_resume = torch.load(opt.retrain)
  local states_resume = torch.load(opt.optimState)
  vae_encoder, vae_decoder, gan = nil, nil, nil
  vae_encoder, vae_decoder, gan = table.unpack(models_resume)
  optimStateVae_encoder, optimStateVae_decoder, optimStateGan = table.unpack(states_resume)
  collectgarbage()
else
  optimStateVae_encoder = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStateVae_decoder = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStateGan = { learningRate = opt.LR/2, optimize = true, numUpdates = 0}
end

-- model parameters and gradient parameters
parametersVae_encoder, gradParametersVae_encoder = vae_encoder:getParameters()
parametersVae_decoder, gradParametersVae_decoder = vae_decoder:getParameters()
parametersGan, gradParametersGan= gan:getParameters()

-- fixed latent variable for validation
local mean = torch.zeros(opt.batchSize, opt.nf * opt.latentDims[1] * opt.latentDims[1]):cuda()
local log_var = torch.zeros(opt.batchSize, opt.nf * opt.latentDims[1] * opt.latentDims[1]):cuda()
z_val_fixed = sampling_z:forward({mean, log_var}):clone()
mean, log_var = nil, nil

-- train VAE-GAN
function train()
  vae_encoder:training()
  vae_decoder:training()
  gan:training()
  epoch = epoch or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local indices = torch.randperm(data.train:size(1)):long():split(opt.batchSize)
  indices[#indices] = nil
  local size = #indices
  local tic = torch.tic()
  local err_vae_encoder_total, err_vae_decoder_total, err_gan_total = 0, 0, 0
  local N = 0
  local reconstruction
  local inputs
  for t,v in ipairs(indices) do
    N = N + 1
    local timer = torch.Timer()

    -- load data and augmentation (horizontal flip)
    local inputs_original = data.train:index(1,v)
    inputs = torch.Tensor(inputs_original:size(1), 3, opt.scales[1], opt.scales[1])
    for i = 1, inputs:size(1) do
      inputs[i] = opt.preprocess_train(image.scale(inputs_original[i], opt.scales[1], opt.scales[1]))
    end
    inputs = inputs:cuda()

    local reconstruction_sample
    local z_sample
    local latent_z
    local df_do

    --[[  update from reconstruction
          forward pass: vae_encoder -> sampling_z -> vae_decoder -> gan
          backward pass: gan -> vae_decoder + data recon -> sampling_z + KLD -> vae_encoder
    --]]
    -- update vae_encoder
    local f_vae_encoder = function(x)
      collectgarbage()
      if x ~= parametersVae_encoder then parametersVae_encoder:copy(x) end
      vae_encoder:zeroGradParameters();
      local output_mean_log_var = vae_encoder:forward(inputs);
      local mean = output_mean_log_var[1]
      local log_var = output_mean_log_var[2]
      latent_z = sampling_z(output_mean_log_var):clone()
      reconstruction = vae_decoder:forward(latent_z):clone()
      local KLDerr = KLD:forward(mean, log_var)
      local dKLD_dtheta = KLD:backward(mean, log_var)
      dKLD_dtheta[1]:mul(opt.alpha)
      dKLD_dtheta[2]:mul(opt.alpha)
      Dislikerr = ReconCriterion:forward(reconstruction, inputs)
      df_do = ReconCriterion:backward(reconstruction, inputs)
      local df_ddecoder = vae_decoder:updateGradInput(latent_z, df_do)
      local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, df_ddecoder)
      vae_encoder:backward(inputs,{df_dsampler[1] + dKLD_dtheta[1], df_dsampler[2] + dKLD_dtheta[2]});
      local Totalerr = Dislikerr + KLDerr * opt.alpha
      return Totalerr, gradParametersVae_encoder
    end
    _, err_vae_encoder = optim[opt.optimization](f_vae_encoder, parametersVae_encoder, optimStateVae_encoder)
    err_vae_encoder_total = err_vae_encoder_total + err_vae_encoder[1]

    -- update vae_decoder
    local f_vae_decoder = function(x)
      if x ~= parametersVae_decoder then parametersVae_decoder:copy(x) end
      vae_decoder:zeroGradParameters();
      local label_gan = torch.ones(opt.batchSize):cuda()
      gan:forward(reconstruction)
      local gan_err = BCECriterion:forward(gan.output, label_gan)
      BCECriterion:backward(gan.output, label_gan)
      local gan_recon = gan:updateGradInput(reconstruction, BCECriterion.gradInput)
      vae_decoder:backward(latent_z, gan_recon * opt.beta  + df_do)
      local mean_sample = torch.zeros(opt.batchSize, sampling_z.output:size(2)):cuda()
      local log_var_sample = torch.zeros(opt.batchSize, sampling_z.output:size(2)):cuda()
      z_sample = sampling_z:forward({mean_sample, log_var_sample}):clone()
      reconstruction_sample = vae_decoder:forward(z_sample):clone()
      gan:forward(reconstruction_sample)
      local label_sample = torch.ones(opt.batchSize):cuda()
      local gan_err_sample = BCECriterion:forward(gan.output, label_sample)
      BCECriterion:backward(gan.output, label_sample)
      local gan_sample = gan:updateGradInput(reconstruction_sample, BCECriterion.gradInput)
      vae_decoder:backward(z_sample, gan_sample * opt.beta)
      local Totalerr = Dislikerr + gan_err * opt.beta + gan_err_sample * opt.beta
      return Totalerr, gradParametersVae_decoder
    end
    _, err_vae_decoder = optim[opt.optimization](f_vae_decoder, parametersVae_decoder, optimStateVae_decoder)
    err_vae_decoder_total = err_vae_decoder_total + err_vae_decoder[1]

    -- update gan
    local f_gan = function(x)
      vae_decoder:evaluate()
      if x ~= parametersGan then parametersGan:copy(x) end
      gan:zeroGradParameters();
      local label_gan = torch.ones(opt.batchSize)
      local inputs_gan = inputs:clone()
      for index = 1, opt.batchSize do
          if index % opt.fakeLabel == 0 then
            label_gan[index] = 0
            inputs_gan[index] = reconstruction[index]:clone()
          end
          if (index+1) % opt.fakeLabel == 0 then
            label_gan[index] = 0
            inputs_gan[index] = reconstruction_sample[index]:clone()
          end
      end
      inputs_gan = inputs_gan:cuda()
      label_gan = label_gan:cuda()
      local outputs_gan = gan:forward(inputs_gan):clone()
      local gan_err = BCECriterion:forward(gan.output, label_gan)
      BCECriterion:backward(gan.output, label_gan)
      gan:backward(inputs_gan, BCECriterion.gradInput);
      local top1 = 0
      local predictions = torch.round(outputs_gan)
      predictions = predictions:float()
      label_gan = label_gan:float()
      for j = 1, outputs_gan:size(1) do
        if label_gan[j] == predictions[j][1] then
          top1 = top1 + 1
        end
      end
      local gan_erate = 1 - top1/outputs_gan:size(1)
      optimStateGan.optimize = true
      if gan_erate < opt.margin then
        print('not training gan, current discriminator error rate is ' .. gan_erate)
        optimStateGan.optimize = false
      end
      print('current discriminator error rate is ' .. gan_erate)
      return gan_err, gradParametersGan
    end
    _, err_gan = optim[opt.optimization .. '_gan'](f_gan, parametersGan, optimStateGan)
    err_gan_total = err_gan_total + err_gan[1]

    print((' | Train: [%d][%d/%d]    Time %.3f  encoder %7.3f (%7.3f)  decoder %7.3f (%7.3f) gan %7.3f (%7.3f)'):format(
         epoch, t, size, timer:time().real,  err_vae_encoder[1], err_vae_encoder_total /N, err_vae_decoder[1], err_vae_decoder_total /N, err_gan[1], err_gan_total /N))
    latent_z = nil
    reconstruction_sample = nil
    z_sample = nil
    Dislikerr = nil
    timer:reset()
    collectgarbage()
  end
  if epoch == 1 then
    image.save(opt.save .. 'original.png', image.toDisplayTensor(inputs:add(1):mul(0.5)))
  end
  image.save(opt.save .. 'recon_' .. epoch .. '_LR_' .. opt.LR .. '_alpha_' .. opt.alpha .. '_beta_' .. opt.beta .. '.png', image.toDisplayTensor(reconstruction:add(1):mul(0.5))) 
  print(('Train loss (vae encoder, vae decoder, gan encoder, gan: '..'%.2f ,'..'%.2f ,' ..'%.2f ,'):format(
        err_vae_encoder_total/N, err_vae_decoder_total/N, err_gan_total/N))
end

function val()
    vae_encoder:evaluate()
    vae_decoder:evaluate()
    gan:evaluate()
    local reconstruction_val = vae_decoder:forward(z_val_fixed)
    reconstruction_val = reconstruction_val:float()
    reconstruction_val:add(1):mul(0.5);
    image.save( opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR .. '_alpha_' .. opt.alpha .. '_beta_' .. opt.beta .. '.png', image.toDisplayTensor(reconstruction_val))
    parametersVae_encoder, gradParametersVae_encoder  = nil, nil 
    parametersVae_decoder, gradParametersVae_decoder  = nil, nil
    parametersGan, gradParametersGan = nil, nil
    if epoch % opt.epochStep == 0 then
       torch.save(opt.save .. 'models_' .. epoch .. '.t7', {vae_encoder, vae_decoder, gan})
       torch.save(opt.save .. 'states_' .. epoch .. '.t7', {optimStateVae_encoder, optimStateVae_decoder, optimStateGan})
    end
    if epoch % opt.step == 0 then
       optimStateVae_encoder.learningRate = optimStateVae_encoder.learningRate * opt.decayLR
       optimStateVae_decoder.learningRate = optimStateVae_decoder.learningRate * opt.decayLR
       optimStateGan.learningRate = optimStateGan.learningRate * opt.decayLR
    end
    parametersVae_encoder, gradParametersVae_encoder = vae_encoder:getParameters()
    parametersVae_decoder, gradParametersVae_decoder = vae_decoder:getParameters()
    parametersGan, gradParametersGan = gan:getParameters()
    print('Saved image to: ' .. opt.save)
end
