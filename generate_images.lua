package.path = package.path .. ';?/?.lua;./?.lua' -- FB
require 'torch';
require 'nn';
require 'optim';
require 'image';
require 'pl';
require 'paths';;
require 'cudnn';
require 'Sampler';
image_utils = require 'image_utils';
require 'stn'

--Set up hyperparameters
opt = {
   modelDir  = '/var/scratch/wshang/crVAE_models/pretrained',    --dir to models
   modelFile = 'birds_Stage1_baseline.t7',                       --model name
   saveDir   = '/var/scratch/wshang/crVAE_samples/birds1/',      --dir to save generations
   nSamples  = 128,                                              --number of samples 
   manualSeed= 424,                                              --random seed
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)


--Parse file name--
if string.find(opt.modelFile, 'baseline') then
    opt.netType = 'baseline'
else
    opt.netType = 'crvae'
end

if string.find(opt.modelFile, 'birds') then
    opt.dataset = 'birds'
elseif string.find(opt.modelFile, 'celeba') then
    opt.dataset = 'celeba'
else 
    opt.dataset = 'lsun'
end

if string.find(opt.modelFile, 'Stage1') then
    opt.stage = 1
    opt.modelStage1 = opt.modelDir .. '/' .. opt.modelFile
else
    opt.stage = 2
    opt.modelStage1 = opt.modelDir .. '/' .. opt.dataset .. '_Stage1_' .. opt.netType .. '.t7'
    opt.modelStage2 = opt.modelDir .. '/' .. opt.modelFile
end

--Load trained 1st stage model--
if opt.netType == 'baseline' then
    models_resume = torch.load(opt.modelStage1)
    vae_encoder, vae_decoder, gan = table.unpack(models_resume)
    gan = nil
    vae_encoder:evaluate()
    vae_decoder:evaluate()
    collectgarbage()
else
    models_resume = torch.load(opt.modelStage1)
    vae_encoder, vae_decoder, var_encoder, var_decoder, gan = table.unpack(models_resume)
    gan = nil
    vae_encoder:evaluate()
    vae_decoder:evaluate()
    var_encoder:evaluate()
    var_decoder:evaluate()
    collectgarbage()
end

if opt.stage == 2 then
    models_up = torch.load(opt.modelStage2)
    models_up[2] = nil
    collectgarbage()
    if opt.dataset == 'lsun' then
        generator = models_up[1]
    else
        generator = nn.Sequential():add(nn.GPU(models_up[1], 2, 1))
        generator:cuda()
    end
    models_up = nil
    collectgarbage()
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

print('Saving everything to: ' .. opt.saveDir)
os.execute('mkdir -p ' .. opt.saveDir)


sampling_z = nn.Sampler()
sampling_z:cuda()
N = 0 
iter = torch.ceil(opt.nSamples/128)
    


for t = 1, iter do 
    local reconstruction_sample
    if opt.netType == 'baseline' then
        local mean_sample = torch.zeros(128, 1024):cuda()
        local log_var_sample = torch.zeros(128, 1024):cuda()
        local z_sample = sampling_z:forward({mean_sample, log_var_sample})
        vae_decoder:forward(z_sample);
        reconstruction_sample = vae_decoder.output
    else
        local mean_sample = torch.zeros(128, 64,4,4):cuda()
        local log_var_sample = torch.zeros(128, 64,4,4):cuda()
        local z_sample = sampling_z:forward({mean_sample, log_var_sample})
        var_decoder:forward(z_sample);
        vae_decoder:forward(var_decoder.output);
        reconstruction_sample = vae_decoder.output
    end
    collectgarbage()
    if opt.stage == 1 then
    	  for i = 1,128 do 
    	      if N < opt.nSamples then
    	          N = N + 1
    	          input = reconstruction_sample[i]:float()
    	          image.save(opt.saveDir .. '/gen_' .. N .. '.png', image.toDisplayTensor(input:add(1):mul(0.5)))
    	      end
        end
	  else 
        reconstruction_sample = generator:forward(vae_decoder.output[{{1,64},{},{},{}}]);
        for i = 1, 64 do 
            if N < opt.nSamples then
                N = N + 1
                input = reconstruction_sample[i]:float()
                image.save(opt.saveDir .. '/gen_' .. N .. '.png', image.toDisplayTensor(input:add(1):mul(0.5)))
            end
        end
        reconstruction_sample = nil
        collectgarbage()
        collectgarbage()
        reconstruction_sample = generator:forward(vae_decoder.output[{{65,128},{},{},{}}]);
        for i = 1, 64 do 
            if N < opt.nSamples then
                N = N + 1
                input = reconstruction_sample[i]:float()
                image.save(opt.saveDir .. '/gen_' .. N .. '.png', image.toDisplayTensor(input:add(1):mul(0.5)))
            end
        end
        reconstruction_sample = nil
        collectgarbage()
        collectgarbage()
    end
end