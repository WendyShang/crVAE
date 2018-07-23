optim = require 'optim'
require 'adam_gan'

-- load models and training criterions
local vae_encoder, vae_decoder, var_encoder, var_decoder, gan, recon, gan_feature, sampling_z = table.unpack(models)
local KLD, ReconCriterion, BCECriterion, ReconZCriterion = table.unpack(criterions)
local sampling_z2 = sampling_z:clone()
local latent_division = opt.nf/opt.timeStep
local optimStateVae_encoder, optimStateVae_decoder, optimStateVar_encoder, optimStateVar_decoder, optimStateGan, optimStateRecon, optimStateFeature

-- reload if previous checkpoint exists
-- otherwise initialize optimStates 
if opt.retrain ~= 'none' then
  models_resume = torch.load(opt.retrain)
  states_resume = torch.load(opt.optimState)
  vae_encoder, vae_decoder, var_encoder, var_decoder, gan, recon, gan_feature = nil, nil, nil, nil, nil, nil, nil
  vae_encoder, vae_decoder, var_encoder, var_decoder, gan, recon, gan_feature = table.unpack(models_resume)
  optimStateVae_encoder, optimStateVae_decoder, optimStateVar_encoder, optimStateVar_decoder, optimStateGan, optimStateRecon, optimStateFeature = table.unpack(states_resume)
  collectgarbage()
else
  optimStateVae_encoder = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStateVae_decoder = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStateGan = { learningRate = opt.LR, optimize = true, numUpdates = 0, step = opt.discRatio}
  optimStateRecon = { learningRate = opt.LR, optimize = true, numUpdates = 0}
  optimStateFeature = { learningRate = opt.LR/2, optimize = true, numUpdates = 0}
  optimStateVar_encoder = { learningRate = opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
  optimStateVar_decoder = { learningRate = opt.LR/opt.timeStep, optimize = true, numUpdates = 0}
end

-- model parameters and gradient parameters
parametersVae_encoder, gradParametersVae_encoder = vae_encoder:getParameters()
parametersVae_decoder, gradParametersVae_decoder = vae_decoder:getParameters()
parametersVar_encoder, gradParametersVar_encoder = var_encoder:getParameters()
parametersVar_decoder, gradParametersVar_decoder = var_decoder:getParameters()
parametersGan, gradParametersGan= gan:getParameters()
parametersRecon, gradParametersRecon= recon:getParameters()
parametersFeature, gradParametersFeature= gan_feature:getParameters()

-- fixed latent variable for validation
local mean = torch.zeros(opt.batchSize, opt.nf, opt.latentDims[1], opt.latentDims[1]):cuda()
local log_var = torch.zeros(opt.batchSize, opt.nf, opt.latentDims[1], opt.latentDims[1]):cuda()
local z_val_fixed = sampling_z:forward({mean, log_var}):clone()
mean,log_var = nil, nil

-- train crVAE-GAN
function train(opt)
  vae_encoder:training()
  vae_decoder:training()
  var_encoder:training()
  var_decoder:training()
  gan:training()
  recon:training()
  gan_feature:training()
  epoch = epoch or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local indices = torch.randperm(data.train:size(1)):long():split(opt.batchSize)
  indices[#indices] = nil
  local size = #indices
  local tic = torch.tic()
  local KLD_total, Recon_total, ReconZ_total = 0, 0, 0
  local N = 0
  local reconstruction
  inputs = torch.Tensor(opt.batchSize, 3, opt.scales[1], opt.scales[1])
  for t,v in ipairs(indices) do
    N = N + 1
    local timer = torch.Timer()

    -- load data and augmentation (horizontal flip)
    local inputs_original = data.train:index(1,v)
    N = N + 1
    for i = 1, inputs:size(1) do
      inputs[i] = opt.preprocess_train(image.scale(inputs_original[i], opt.scales[1], opt.scales[1]))
    end
    inputs = inputs:cuda()

    --[[  update from reconstruction: forward pass
          vae_encoder -> var_encoder (LSTM) -> sampling_z 
          -> var_decoder (LSTM) -> vae_decoder -> gan_feature 
          -> latent variable recon (a.k.a. MI max.) or gan (discriminator)
    --]]
    local output_mean_log_var_before = vae_encoder:forward(inputs);
    local output_mean_log_var = var_encoder:forward(output_mean_log_var_before)
    local latent_z = sampling_z:forward(output_mean_log_var)
    local output_decoder_before = var_decoder:forward(latent_z)
    reconstruction = vae_decoder:forward(output_decoder_before);
    local feature = gan_feature:forward(reconstruction)
    local z_recon = recon:forward(feature)
    local gan_recon = gan:forward(feature)

    --[[  update from reconstruction: backward pass
          latent variable recon, gan -> gan_feature + data recon 
          -> vae_decoder + latent variable recon
          -> var_decoder -> sampling_z + KLD -> var_encoder -> vae_encoder
    --]]
    reconstruction = vae_decoder.output:clone()
    local mean = output_mean_log_var[1]
    local log_var = output_mean_log_var[2]
    local KLDerr = KLD:forward(mean, log_var)
    local Dislikerr = ReconCriterion:forward(reconstruction, inputs)
    local ZDislikerr = ReconZCriterion:forward(z_recon, output_decoder_before)
    local dKLD_dtheta = KLD:backward(mean, log_var)
    local df_do = ReconCriterion:backward(reconstruction, inputs)
    local df_dz_recon = ReconZCriterion:backward(z_recon, output_decoder_before)
    local label_gan = torch.ones(opt.batchSize):cuda()
    KLD_total = KLD_total + KLDerr
    ReconZ_total = ReconZ_total + ZDislikerr*0.5
    Recon_total = Recon_total + Dislikerr

    -- update recon 
    local f_recon_z = function(x)
      if x ~= parametersRecon then parametersRecon:copy(x) end
      recon:zeroGradParameters()
      df_dz_recon[{{},{1, latent_division},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division+1, latent_division*3},{},{}}]:mul(1.5);
      df_dz_recon[{{},{latent_division*3+1, latent_division*4},{},{}}]:mul(1.25);
      df_dz_recon[{{},{latent_division*4+1, latent_division*6},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division*6+1, latent_division*8},{},{}}]:mul(1);
      recon:backward(feature, df_dz_recon)
      local Totalerr = ZDislikerr
      return Totalerr, gradParametersRecon
    end
    _, err_z_recon = optim[opt.optimization](f_recon_z, parametersRecon, optimStateRecon)

    -- update gan_feature (backprop from latent space reconstruction)
    local f_recon_feature = function(x)
      if x ~= parametersFeature then parametersFeature:copy(x) end
      gan_feature:zeroGradParameters()
      gan_feature:backward(reconstruction, recon.gradInput)
      local Totalerr = ZDislikerr
      return Totalerr, gradParametersFeature
    end
    _, err_feature_recon = optim[opt.optimization](f_recon_feature, parametersFeature, optimStateFeature)
    local df_recon_feature = gan_feature.gradInput:clone()

    -- update vae_decoder (backprop from recon, gan, input reconstruction)
    local f_vae_decoder = function(x)
      if x ~= parametersVae_decoder then parametersVae_decoder:copy(x) end
      vae_decoder:zeroGradParameters()
      gan_feature:forward(vae_decoder.output)
      gan:forward(gan_feature.output)
      local gan_err = BCECriterion:forward(gan.output, label_gan)
      BCECriterion:backward(gan.output, label_gan)
      gan:updateGradInput(gan_feature.output, BCECriterion.gradInput)
      gan_feature:updateGradInput(vae_decoder.output, gan.gradInput)
      vae_decoder:backward(var_decoder.output, df_do + df_recon_feature*opt.kappa + gan_feature.gradInput*opt.beta);
      local Totalerr = Dislikerr + gan_err
      return Totalerr, gradParametersVae_decoder
    end
    _, err_vae_decoder = optim[opt.optimization](f_vae_decoder, parametersVae_decoder, optimStateVae_decoder)
    df_recon_feature = nil

    -- update var_decoder (backprop from vae_decoder)
    local f_var_decoder = function(x)
      if x ~= parametersVar_decoder then parametersVar_decoder:copy(x) end
      var_decoder:zeroGradParameters()
      var_decoder:backward(latent_z, vae_decoder.gradInput);
      if opt.grad_clip > 0 then
         gradParametersVar_decoder:clamp(-opt.grad_clip, opt.grad_clip)
      end
      local Totalerr = Dislikerr
      return Totalerr, gradParametersVar_decoder
    end
    _, err_var_decoder = optim[opt.optimization](f_var_decoder, parametersVar_decoder, optimStateVar_decoder)

    -- update var_encoder (backprop from var_decoder and KLD, KL weighting applied)
    local f_var_encoder = function(x)
      if x ~= parametersVar_encoder then parametersVar_encoder:copy(x) end
      var_encoder:zeroGradParameters()
      local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, var_decoder.gradInput)
      assert(opt.timeStep == 8, 'currently only support time step 8')
      dKLD_dtheta[1][{{},{1, latent_division*3},{},{}}]:mul(opt.alpha1)
      dKLD_dtheta[2][{{},{1, latent_division*3},{},{}}]:mul(opt.alpha1)
      dKLD_dtheta[1][{{},{latent_division*3+1, latent_division*8},{},{}}]:mul(opt.alpha2)
      dKLD_dtheta[2][{{},{latent_division*3+1, latent_division*8},{},{}}]:mul(opt.alpha2)
      var_encoder:backward(output_mean_log_var_before, {dKLD_dtheta[1] + df_dsampler[1], dKLD_dtheta[2] + df_dsampler[2]})
      if opt.grad_clip > 0 then
         gradParametersVar_encoder:clamp(-opt.grad_clip, opt.grad_clip)
      end
      local Totalerr = Dislikerr + KLDerr * (opt.alpha1+opt.alpha2) * 0.5
      return Totalerr, gradParametersVar_encoder
    end
    _, err_var_encoder = optim[opt.optimization](f_var_encoder, parametersVar_encoder, optimStateVar_encoder)

    -- update vae_encoder (backprop from var_encoder)
    local f_vae_encoder = function(x)
      if x ~= parametersVae_encoder then parametersVae_encoder:copy(x) end
    	vae_encoder:zeroGradParameters()
      vae_encoder:backward(inputs, var_encoder.gradInput);
    	local Totalerr = Dislikerr + KLDerr * (opt.alpha1+opt.alpha2) * 0.5
    	return Totalerr, gradParametersVae_encoder
	  end
    _, err_vae_encoder = optim[opt.optimization](f_vae_encoder, parametersVae_encoder, optimStateVae_encoder)

    --[[  update from generation: forward pass
          sampling_z -> var_decoder (LSTM) -> vae_decoder -> gan_feature 
          -> latent variable recon (a.k.a. MI max.) or gan (discriminator)
    --]]
    local mean_sample = torch.zeros(opt.batchSize, sampling_z.output:size(2),sampling_z.output:size(3),sampling_z.output:size(4)):cuda()
    local log_var_sample = torch.zeros(opt.batchSize, sampling_z.output:size(2),sampling_z.output:size(3),sampling_z.output:size(4)):cuda()
    local z_sample = sampling_z2:forward({mean_sample, log_var_sample})
    local output_decoder_before_sample = var_decoder:forward(z_sample)
    vae_decoder:forward(output_decoder_before_sample)

    --[[  update from generation: backward pass
          latent variable recon, gan -> gan_feature
          -> vae_decoder + latent variable recon -> var_decoder -> sampling_z
    --]]
    reconstruction_sample = vae_decoder.output:clone()
    local feature_sample = gan_feature:forward(reconstruction_sample)
    local z_recon_sample = recon:forward(feature_sample)
    local ZDislikerr_sample = ReconZCriterion:forward(z_recon_sample, output_decoder_before_sample)
    local df_dz_recon_sample = ReconZCriterion:backward(z_recon_sample, output_decoder_before_sample)
    ReconZ_total = ReconZ_total + ZDislikerr_sample*0.5

    -- update recon
    local f_recon_z_sample = function(x)
      if x ~= parametersRecon then parametersRecon:copy(x) end
      recon:zeroGradParameters()
      df_dz_recon[{{},{1, latent_division},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division+1, latent_division*3},{},{}}]:mul(1.5);
      df_dz_recon[{{},{latent_division*3+1, latent_division*4},{},{}}]:mul(1.25);
      df_dz_recon[{{},{latent_division*4+1, latent_division*6},{},{}}]:mul(0.1);
      df_dz_recon[{{},{latent_division*6+1, latent_division*8},{},{}}]:mul(1);
      recon:backward(feature_sample, df_dz_recon_sample)
      local Totalerr = ZDislikerr_sample
      return Totalerr, gradParametersRecon
    end
    _, err_z_recon_sample = optim[opt.optimization](f_recon_z_sample, parametersRecon, optimStateRecon)

    -- update gan_feature (backprop from latent space reconstruction)
    local f_recon_feature_sample = function(x)
      if x ~= parametersFeature then parametersFeature:copy(x) end
      gan_feature:zeroGradParameters()
      gan_feature:backward(reconstruction_sample, recon.gradInput)
      local Totalerr = ZDislikerr_sample
      return Totalerr, gradParametersFeature
    end
    _, err_feature_recon_sample = optim[opt.optimization](f_recon_feature_sample, parametersFeature, optimStateFeature)
    local df_recon_feature_sample = gan_feature.gradInput:clone()

    -- update vae_decoder (backprop from recon and gan)
    local f_vae_decoder_sample = function(x)
      if x ~= parametersVae_decoder then parametersVae_decoder:copy(x) end
      vae_decoder:zeroGradParameters()
      gan_feature:forward(vae_decoder.output)
      gan:forward(gan_feature.output)
      local gan_err = BCECriterion:forward(gan.output, label_gan)
      BCECriterion:backward(gan.output, label_gan)
      gan:updateGradInput(gan_feature.output, BCECriterion.gradInput)
      gan_feature:updateGradInput(vae_decoder.output, gan.gradInput)
      vae_decoder:backward(var_decoder.output, gan_feature.gradInput*opt.kappa + df_recon_feature_sample*opt.beta);
      local Totalerr = gan_err + ZDislikerr_sample
      return Totalerr, gradParametersVae_decoder
    end
    _, err_vae_decoder_sample = optim[opt.optimization](f_vae_decoder_sample, parametersVae_decoder, optimStateVae_decoder)
    df_recon_feature_sample = nil

    -- update var_decoder (backprop from vae_decoder)
    local f_var_decoder_sample = function(x)
      if x ~= parametersVar_decoder then parametersVar_decoder:copy(x) end
      var_decoder:zeroGradParameters()
      var_decoder:backward(z_sample, vae_decoder.gradInput);
      if opt.grad_clip > 0 then
         gradParametersVar_decoder:clamp(-opt.grad_clip, opt.grad_clip)
      end
      local Totalerr = ZDislikerr_sample
      return Totalerr, gradParametersVar_decoder
    end
    _, err_var_decoder_sample = optim[opt.optimization](f_var_decoder_sample, parametersVar_decoder, optimStateVar_decoder)

    --[[  update discriminator with half real, 
          quarter fake from reconstruction, quarter fake from generation
          (i.e., opt.fakeLabel = 4)
    --]]
    local inputs_gan = inputs
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
    local outputs_feature = gan_feature:forward(inputs_gan):clone()
    local outputs_gan = gan:forward(outputs_feature)
    local gan_err = BCECriterion:forward(gan.output, label_gan)
    BCECriterion:backward(gan.output, label_gan)

    -- update gan
    local f_gan = function(x)
      if x ~= parametersGan then parametersGan:copy(x) end
      gan:zeroGradParameters()
      gan:backward(outputs_feature, BCECriterion.gradInput);
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

    -- update gan_feature (backprop from gan)
    if optimStateGan.optimize then
        local f_gan_feature = function(x)
          if x ~= parametersFeature then parametersFeature:copy(x) end
          gan_feature:zeroGradParameters()
          gan_feature:backward(inputs_gan, gan.gradInput)
          local Totalerr = gan_err
          return Totalerr, gradParametersFeature
        end
        _, err_feature_gan = optim[opt.optimization](f_gan_feature, parametersFeature, optimStateFeature)
    end

    label_gan = nil
    inputs_gan = nil
    outputs_feature = nil
    outputs_gan = nil
    gan_err = nil

    print((' | Train: [%d][%d/%d]    Time %.3f  KL %7.3f (%7.3f)  recon %7.3f (%7.3f) recon_z %7.3f (%7.3f)'):format(
         epoch, t, size, timer:time().real,  KLDerr, KLD_total/N, Dislikerr, Recon_total/N, ZDislikerr, ReconZ_total/N))

    timer:reset()
    collectgarbage()
  end
  if epoch == 1 then
      image.save(opt.save .. 'original.png', image.toDisplayTensor(inputs:add(1):mul(0.5)))
  end
  image.save(opt.save .. 'recon_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha1 .. opt.alpha2 .. '_beta_' .. opt.beta .. '.png', image.toDisplayTensor(reconstruction:add(1):mul(0.5))) 
  print(('Train loss (KLD, Recon, ReconZ: '..'%.2f ,'..'%.2f ,' ..'%.2f ,'):format(
        KLD_total/N, Recon_total/N, ReconZ_total/N))
end

function val(opt)
  vae_encoder:evaluate()
  vae_decoder:evaluate()
  var_encoder:evaluate()
  var_decoder:evaluate()
  gan_feature:evaluate()
  recon:evaluate()
  var_decoder:forward(z_val_fixed)
  local reconstruction_val = vae_decoder:forward(var_decoder.output)
  reconstruction_val = reconstruction_val:float()
  reconstruction_val:add(1):mul(0.5)
  image.save( opt.save .. 'gen_' .. epoch .. '_LR_' .. opt.LR .. '_alphas_' .. opt.alpha1 .. opt.alpha2 .. '_beta_' .. opt.beta .. '.png', image.toDisplayTensor(reconstruction_val))
  parametersVae_encoder, gradParametersVae_encoder  = nil, nil
  parametersVae_decoder, gradParametersVae_decoder  = nil, nil
  parametersVar_encoder, gradParametersVar_encoder  = nil, nil
  parametersVar_decoder, gradParametersVar_decoder  = nil, nil
  parametersGan, gradParametersGan= nil, nil
  parametersRecon, gradParametersRecon= nil, nil
  parametersFeature, gradParametersFeature= nil, nil
  if epoch % opt.step == 0 then
    optimStateVae_encoder.learningRate = optimStateVae_encoder.learningRate*opt.decayLR
    optimStateVae_decoder.learningRate = optimStateVae_decoder.learningRate*opt.decayLR
    optimStateVar_encoder.learningRate = optimStateVar_encoder.learningRate*opt.decayLR
    optimStateVar_decoder.learningRate = optimStateVar_decoder.learningRate*opt.decayLR
    optimStateGan.learningRate = optimStateGan.learningRate*opt.decayLR
    optimStateRecon.learningRate = optimStateRecon.learningRate*opt.decayLR
    optimStateFeature.learningRate = optimStateFeature.learningRate*opt.decayLR
  end
  parametersVae_encoder, gradParametersVae_encoder = vae_encoder:getParameters()
  parametersVae_decoder, gradParametersVae_decoder = vae_decoder:getParameters()
  parametersVar_encoder, gradParametersVar_encoder = var_encoder:getParameters()
  parametersVar_decoder, gradParametersVar_decoder = var_decoder:getParameters()
  parametersGan, gradParametersGan= gan:getParameters()
  parametersRecon, gradParametersRecon= recon:getParameters()
  parametersFeature, gradParametersFeature= gan_feature:getParameters()
  print('Saved image to: ' .. opt.save)
end
