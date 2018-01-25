optim = require 'optim'

--helper functions
function normpdf(latent)
  return -torch.cmul(latent, latent):add(math.log(2*math.pi)):mul(0.5):sum(2)
end

function gaussianpdf(latent, mean, log_var)
  local var = torch.exp(0.5*log_var)
  local tmp = torch.cdiv(latent-mean, var)
  return -(torch.cmul(tmp, tmp):add(math.log(2*math.pi)) + log_var):mul(0.5):sum(2)
end

local function binarize(data)
  return (data - torch.rand(data:size())):ge(0):float()
end

--load training functions etc.
vae_encoder, vae_decoder, sampling_z = table.unpack(models)
local KLD, ReconCriterion = table.unpack(criterions)
local nSample = opt.nSample

--reload if previous checkpoint exisits
-- otherwise initialize optimStates
if opt.retrain ~= 'none' then
  models_resume = torch.load(opt.retrain)
  states_resume = torch.load(opt.optimState)
  vae_encoder = nil
  vae_decoder = nil
  collectgarbage()
  vae_encoder, vae_decoder = table.unpack(models_resume)
  optimStateVae_encoder, optimStateVae_decoder = table.unpack(states_resume)
else
  optimStateVae_encoder = { learningRate = opt.LR}
  optimStateVae_decoder = { learningRate = opt.LR}
end

-- model parameters and gradient parameters
parametersVae_encoder, gradParametersVae_encoder = vae_encoder:getParameters()
parametersVae_decoder, gradParametersVae_decoder = vae_decoder:getParameters()


function train()
  vae_encoder:training()
  vae_decoder:training()
  epoch = epoch or 1
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local indices = torch.randperm(data.train:size(1)):long():split(opt.batchSize)
  indices[#indices] = nil
  local size = #indices
  local tic = torch.tic()
  local KLD_total, Recon_total = 0, 0
  local N = 0
  for t,v in ipairs(indices) do
    N = N + 1
    local timer = torch.Timer()

    -- load data and augmentation 
    local inputs_original = data.train:index(1,v)
    local inputs = torch.Tensor(inputs_original:size(1), 1, opt.scales[1], opt.scales[1])
    for i = 1, inputs:size(1) do
      inputs[i] = binarize(inputs_original[i])
    end
    inputs  = inputs:cuda()

    --[[  update from reconstruction: forward pass
          vae_encoder ->  sampling_z 
          -> vae_decoder
    --]]
    local output_mean_log_var = vae_encoder:forward(inputs);
    local mean = output_mean_log_var[1]
    local log_var = output_mean_log_var[2]
    local latent_z = sampling_z(output_mean_log_var)
    local reconstruction = vae_decoder:forward(latent_z)
    local KLDerr = KLD:forward(mean, log_var)
    local dKLD_dtheta = KLD:backward(mean, log_var)
    local Dislikerr = ReconCriterion:forward(reconstruction, inputs)
    KLD_total = KLD_total + KLDerr
    Recon_total = Recon_total + Dislikerr
    dKLD_dtheta[1]:mul(opt.alpha)
    dKLD_dtheta[2]:mul(opt.alpha)
    local df_do = ReconCriterion:backward(reconstruction, inputs)
    local df_ddecoder = vae_decoder:updateGradInput(latent_z, df_do)
    local df_dsampler = sampling_z:updateGradInput(output_mean_log_var, df_ddecoder)
    local f_vae_encoder = function(x)
      collectgarbage()
      if x ~= parametersVae_encoder then parametersVae_encoder:copy(x) end
      vae_encoder:zeroGradParameters()
    	vae_encoder:backward(inputs,{df_dsampler[1]+dKLD_dtheta[1], df_dsampler[2]+dKLD_dtheta[2]})
    	local Totalerr = Dislikerr + KLDerr * opt.alpha
    	return Totalerr, gradParametersVae_encoder
	  end
    _, _ = optim[opt.optimization](f_vae_encoder, parametersVae_encoder, optimStateVae_encoder)

    local f_vae_decoder = function(x)
    	if x ~= parametersVae_decoder then parametersVae_decoder:copy(x) end
    	vae_decoder:zeroGradParameters()
    	vae_decoder:backward(latent_z, df_do);
    	local Totalerr = Dislikerr
    	return Totalerr, gradParametersVae_decoder
	  end
    _, _ = optim[opt.optimization](f_vae_decoder, parametersVae_decoder, optimStateVae_decoder)
    
    print((' | Train: [%d][%d/%d]    Time %.3f  recon %7.3f (%7.3f)  KLD %7.3f (%7.3f)'):format(
         epoch, t, size, timer:time().real,  Dislikerr, Recon_total /N, KLDerr, KLD_total /N))
    timer:reset()
    collectgarbage()
  end

  print(('Train loss (Recon, KLD): '..'%.2f ,'..'%.2f ,'):format(Recon_total/N, KLD_total/N)) 
end

function val()
  vae_encoder:evaluate()
  vae_decoder:evaluate()
  local batchSize = opt.batchSize
  local ll_all = torch.Tensor(math.floor(data.test:size(1)/batchSize)*batchSize)
  for n = 1,math.floor(data.test:size(1)/batchSize) do
    local idx = {{(n-1)*batchSize+1,n*batchSize}}
    local input = data.test[idx]
    local output_mean_log_var = vae_encoder:forward(input:cuda())
    local mean = output_mean_log_var[1]
    local log_var = output_mean_log_var[2]
    local out = torch.Tensor(nSample, batchSize):cuda()
    for i = 1,nSample do
      local latent = sampling_z(output_mean_log_var)
      local recon = vae_decoder:forward(latent)
      recon:clamp(1e-10, 1-1e-10)
      local ll = torch.log(recon):cmul(input:cuda()):sum(2):sum(3):sum(4):squeeze() + torch.log(1-recon):cmul((1-input):cuda()):sum(2):sum(3):sum(4):squeeze()
      local ll_prior = normpdf(latent):squeeze()
      local ll_rec = gaussianpdf(latent, mean, log_var):squeeze()
      out[i] = ll + ll_prior - ll_rec
    end
    collectgarbage()
    ll_all[idx] = (torch.log(torch.exp(out - torch.max(out, 1):expand(nSample,batchSize)):sum(1)) + torch.max(out, 1)):add(-math.log(nSample)):squeeze():float()
  end
  print(string.format('Estimated NLL  %1.4f', ll_all:mean()))
  parametersVae_encoder, gradParametersVae_encoder  = nil, nil -- nil them to avoid spiking memory
  parametersVae_decoder, gradParametersVae_decoder  = nil, nil
  if epoch % opt.epochStep == 0 then
    torch.save(opt.save .. 'models_' .. epoch .. '.t7', {vae_encoder:clearState(), vae_decoder:clearState()})
    torch.save(opt.save .. 'states_' .. epoch .. '.t7', {optimStateVae_encoder, optimStateVae_decoder})
  end
  if epoch % opt.step == 0 then
    optimStateVae_encoder.learningRate = optimStateVae_encoder.learningRate/2
    optimStateVae_decoder.learningRate = optimStateVae_decoder.learningRate/2
  end
  parametersVae_encoder, gradParametersVae_encoder = vae_encoder:getParameters()
  parametersVae_decoder, gradParametersVae_decoder = vae_decoder:getParameters()
  return ll_all:mean()
end


function evaluate(opt, nSample)
  local nSample = nSample
  local models = torch.load(opt.save .. '/model_best.t7')
  local vae_encoder, vae_decoder = table.unpack(models)
  vae_encoder:cuda();
  vae_decoder:evaluate();
  vae_decoder:cuda();
  local batchSize = opt.batchSize
  local ll_all = torch.Tensor(math.floor(data.test:size(1)/batchSize)*batchSize)
  local vlb_all = torch.Tensor(math.floor(data.test:size(1)/batchSize)*batchSize)
  for n = 1,math.floor(data.test:size(1)/batchSize) do
    local idx = {{(n-1)*batchSize+1,n*batchSize}}
    local input = data.test[idx]
    local output_mean_log_var = vae_encoder:forward(input:cuda())
    local mean = output_mean_log_var[1]
    local log_var = output_mean_log_var[2]
    local out = torch.Tensor(nSample, batchSize):cuda()
    for i = 1,nSample do
      local latent = sampling_z(output_mean_log_var)
      local recon = vae_decoder:forward(latent)
      recon:clamp(1e-5, 1-1e-5)
      local ll = torch.log(recon):cmul(input:cuda()):sum(2):sum(3):sum(4):squeeze()
      ll = ll + torch.log(1-recon):cmul((1-input):cuda()):sum(2):sum(3):sum(4):squeeze()
      local ll_prior = normpdf(latent):squeeze()
      local ll_rec = gaussianpdf(latent, mean, log_var):squeeze()
      out[i] = ll + ll_prior - ll_rec
    end
    ll_all[idx] = (torch.log(torch.exp(out - torch.max(out, 1):expand(nSample,batchSize)):sum(1)) + torch.max(out, 1)):add(-math.log(nSample)):squeeze():float()
    vlb_all[idx] = out:mean(1)[1]:float()
  end
  return ll_all:mean(), vlb_all:mean()
end
