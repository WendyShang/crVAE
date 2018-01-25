local function ConcatAct()
   local model = nn.ConcatTable()
   model:add(nn.Identity())
   model:add(nn.MulConstant(-1))
   return model
end

local function createModel(opt)
   local encoder = nn.Sequential()
   local decoder = nn.Sequential()
   local var_encoder = nn.ParallelTable()
   local var_decoder = nn.Sequential()
   local gan = nn.Sequential()
   local gan_feature = nn.Sequential()
   local recon = nn.Sequential()
   local baseChannels = opt.baseChannels
   local w = opt.latentDims[1]
   local z = opt.nf
   local eps = opt.eps
   local mom = opt.mom
   local time_step = opt.timeStep

   if opt.dataset == 'bird' then
      -----------------------------------
      -- Encoder (Inference network) ----
      -- convolution net -> LSTM layer --
      -----------------------------------
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))

      -- conv1: 64 x 64 --> 32 x 32
      encoder:add(cudnn.SpatialConvolution(6, baseChannels, 5, 5, 2, 2, 2, 2))
      encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv2: 32 x 32 --> 16 x 16
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps,mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv3-1, conv3-2: 16 x 16 --> 8 x 8
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      encoder:add(nn.LeakyReLU(0.1))

      -- conv4-1, conv4-2: 8 x 8 --> 4 x 4
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      encoder:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      encoder:add(nn.LeakyReLU(0.1))

      -- LSTM layer for Channel-Recurrency
      -- mean path: convolution followed by bias subtraction (mu_0 in figure 2(c))
      local mean_shift = nn.Sequential()
      local add_size = torch.LongStorage(3)
      add_size[1] = z
      add_size[2] = w
      add_size[3] = w
      mean_shift:add(cudnn.SpatialConvolution(baseChannels, z, 3, 3, 1, 1, 1, 1))
      mean_shift:add(nn.Add(add_size))
      mean_shift:get(2):reset(0.01)

      local mean_logvar_before = nn.ConcatTable()
      mean_logvar_before:add(mean_shift)
      mean_logvar_before:add(nn.Identity())
      encoder:add(mean_logvar_before)

      -- variance path: 
      -- 1. 4 x 4 x baseChannels is divided into time_step blocks of size 4 x 4 x baseChannels/time_step
      -- 2. channel-recurrency via LSTM followed by block-wise FC layer to generate \sigma^{rnn} in figure 2(c)
      local input_division = baseChannels/time_step
      local latent_division = z/time_step
      local var_transform = nn.Sequential()
      local LSTM_encoder = cudnn.LSTM(w*w*input_division, w*w*input_division, 2, true)
      for i = 0, LSTM_encoder.numLayers - 1 do
           local params = getParams(LSTM_encoder, i, 1)
           params.bias:fill(1)
      end
      var_transform:add(nn.View(time_step, w*w*input_division):setNumInputDims(3))
      var_transform:add(nn.Contiguous())
      var_transform:add(LSTM_encoder)
      var_transform:add(nn.Contiguous())
      var_transform:add(nn.View(opt.batchSize*time_step, w*w*input_division))
      var_transform:add(nn.Contiguous())
      var_transform:add(nn.BatchNormalization(w*w*input_division))
      var_transform:add(nn.Linear(w*w*input_division, w*w*latent_division))
      var_transform:add(nn.View(z, w, w):setNumInputDims(2))
      var_transform:add(nn.Contiguous())

      var_encoder:add(nn.Identity())
      var_encoder:add(var_transform)

      -------------------------------------
      -- Decoder (Generation network)    --
      -- LSTM layer -> deconvolution net --
      -------------------------------------
      -- Channel-Recurrent Decoder that transforms z_{i} into u_{i} in figure 2(c)
      local LSTM_decoder = cudnn.LSTM(w*w*latent_division, w*w*input_division, 2, true)
      for i = 0, LSTM_decoder.numLayers - 1 do
           local params = getParams(LSTM_decoder, i, 1)
           params.bias:fill(1)
      end
      var_decoder:add(nn.View(z/latent_division, w * w * latent_division):setNumInputDims(3))
      var_decoder:add(nn.Contiguous())
      var_decoder:add(LSTM_decoder)
      var_decoder:add(nn.Contiguous())
      var_decoder:add(nn.View(opt.batchSize*z/latent_division, w*w*input_division):setNumInputDims(3))
      var_decoder:add(nn.Contiguous())
      var_decoder:add(nn.Linear(w*w*input_division, w*w*latent_division))
      var_decoder:add(nn.View(opt.batchSize, z, w, w))
      var_decoder:add(nn.Contiguous())

      -- add bias back (figure 2(c))
      local add_size = torch.LongStorage(3)
      add_size[1] = z
      add_size[2] = w
      add_size[3] = w
      decoder:add(nn.Add(add_size))
      decoder:get(1):reset(0.01)

      -- deconv5: 4 x 4 --> 4 x 4
      decoder:add(cudnn.SpatialConvolution(z, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))

      -- deconv4-1, deconv4-2: 4 x 4 --> 8 x 8
      decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      decoder:add(nn.LeakyReLU(0.1))
      decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      decoder:add(nn.LeakyReLU(0.1))
      baseChannels = baseChannels/2

      -- deconv3: 8 x 8 --> 16 x 16
      decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      decoder:add(nn.LeakyReLU(0.1))
      baseChannels = baseChannels/2

      -- deconv2: 16 x 16 --> 32 x 32
      decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      decoder:add(nn.LeakyReLU(0.1))

      -- deconv1: 32 x 32 --> 64 x 64
      decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      decoder:add(nn.LeakyReLU(0.1))

      -- tanH: 64 x 64 --> 64 x 64
      decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      decoder:add(nn.LeakyReLU(0.1))
      decoder:add(cudnn.SpatialConvolution(baseChannels, 3, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.Tanh())

      ------------------------------------------------------
      -- Discriminator network                            --
      -- split to recon (MI max) and gan (discrimination) --
      ------------------------------------------------------
      -- conv1: 64 x 64 --> 32 x 32
      gan_feature:add(cudnn.SpatialConvolution(3, baseChannels, 5, 5, 1, 1, 2, 2))
      gan_feature:add(cudnn.SpatialMaxPooling(2,2))
      gan_feature:add(cudnn.ReLU(true))
      gan_feature:add(nn.SpatialDropout(0.5, true))

      -- conv2: 32 x 32 --> 16 x 16
      gan_feature:add(cudnn.SpatialConvolution(baseChannels, baseChannels*2, 5, 5, 1, 1, 2, 2))
      gan_feature:add(cudnn.SpatialMaxPooling(2,2))
      gan_feature:add(cudnn.ReLU(true))
      gan_feature:add(nn.SpatialDropout(0.5, true))

      -- conv3: 16 x 16 --> 8 x 8
      gan_feature:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels*4, 5, 5, 1, 1, 2, 2))
      gan_feature:add(cudnn.SpatialMaxPooling(2,2))
      gan_feature:add(cudnn.ReLU(true))
      gan_feature:add(nn.SpatialDropout(0.5, true))

      -- conv4: 8 x 8 --> 4 x 4
      gan_feature:add(cudnn.SpatialConvolution(baseChannels*4, baseChannels*4, 3, 3, 1, 1, 1, 1))
      gan_feature:add(cudnn.SpatialMaxPooling(2,2))
      gan_feature:add(cudnn.ReLU(true))
      gan_feature:add(nn.SpatialDropout(0.5, true))

      -- latent space reconstruction head (MI maximization)
      recon:add(cudnn.SpatialConvolution(baseChannels*4, baseChannels, 3, 3, 1, 1, 1, 1))
      recon:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
      recon:add(nn.LeakyReLU(0.2, true))
      recon:add(cudnn.SpatialConvolution(baseChannels, z, 1, 1))
      recon:add(nn.SpatialBatchNormalization(z,eps,mom))
      recon:add(nn.LeakyReLU(0.1, true))
      recon:add(nn.View(z*w*w))
      recon:add(nn.Linear(z*w*w, z*w*w))
      recon:add(nn.View(z,w,w))
      local add_size_recon = torch.LongStorage(3)
      add_size_recon[1] = z
      add_size_recon[2] = w
      add_size_recon[3] = w
      recon:add(nn.Add(add_size_recon))
      recon:get(#recon):reset(0.01)

      -- discriminator head
      -- Fully-connected: 4 x 4 x (baseChannels x 8) --> 4 x 4 x (baseChannels x 2)
      -- followed by batch discrimination
      gan:add(nn.View(w*w*baseChannels*4))
      gan:add(nn.Linear(w*w*baseChannels*4, w*w*baseChannels))
      gan:add(nn.Normalize(2))
      gan:add(nn.Dropout(0.5, true))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.BatchDiscrimination(w*w*baseChannels, 100, 5))
      gan:add(nn.Linear(w*w*baseChannels + 100,1))
      gan:add(nn.Sigmoid())
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(encoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(decoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(var_encoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(var_decoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k,v in pairs(gan:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n),math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(encoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k,v in pairs(decoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')

   for k,v in pairs(encoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k,v in pairs(decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k,v in pairs(gan:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k,v in pairs(var_encoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k,v in pairs(var_decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end

   if opt.cudnn == 'deterministic' then
      encoder:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
      decoder:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   sampling_z = nn.Sampler()
   KLD = nn.KLDCriterion()
   BCECriterion = nn.BCECriterion()
   ReconCriterion = nn.MSECriterion()
   ReconZCriterion = nn.MSECriterion()

   encoder:cuda()
   decoder:cuda()
   var_encoder:cuda()
   var_decoder:cuda()
   ReconCriterion:cuda()
   KLD:cuda()
   sampling_z:cuda()
   BCECriterion:cuda()
   gan:cuda()
   gan_feature:cuda()
   recon:cuda()
   ReconZCriterion:cuda()

   return {encoder, decoder, var_encoder, var_decoder,gan, recon, gan_feature, sampling_z}, {KLD, ReconCriterion, BCECriterion, ReconZCriterion}
end

return createModel
