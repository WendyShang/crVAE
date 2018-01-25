local function ConcatAct()
   local model = nn.ConcatTable()
   model:add(nn.Identity())
   model:add(nn.MulConstant(-1))
   return model
end

local function createModel(opt)
   local encoder = nn.Sequential()
   local decoder = nn.Sequential()
   local gan = nn.Sequential()
   local baseChannels = opt.baseChannels
   local w = opt.latentDims[1]
   local z = opt.nf
   local eps = opt.eps
   local mom = opt.mom

   if opt.dataset == 'bird' then
      ---------------------------------
      -- Encoder (Inference network) --
      -- convolution net -> FC layer --
      ---------------------------------
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))

      -- conv1: 64 x 64 --> 32 x 32
      encoder:add(cudnn.SpatialConvolution(6, baseChannels, 5, 5, 2, 2, 2, 2))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, 1e-6, 0.9))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv2: 32 x 32 --> 16 x 16
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, 1e-6, 0.9))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2

      -- conv3-1, conv3-2: 16 x 16 --> 8 x 8
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, 1e-6, 0.9))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      baseChannels = baseChannels * 2
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(nn.LeakyReLU(0.1))

      -- conv4-1, conv4-2: 8 x 8 --> 4 x 4
      encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(ConcatAct())
      encoder:add(nn.JoinTable(2))
      encoder:add(cudnn.ReLU(true))
      encoder:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels, 3, 3, 1, 1, 1, 1))
      encoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      encoder:add(nn.LeakyReLU(0.1))

      -- Fully-connected: 4 x 4 x baseChannels --> 4 x 4 x z
      encoder:add(nn.View(w*w*baseChannels):setNumInputDims(3))
      local mean_logvar = nn.ConcatTable()
      mean_logvar:add(nn.Linear(w*w*baseChannels, w*w*z)) -- mean
      mean_logvar:add(nn.Linear(w*w*baseChannels, w*w*z)) -- variance
      encoder:add(mean_logvar)

      -----------------------------------
      -- Decoder (Generation network)  --
      -- FC layer -> deconvolution net --
      -----------------------------------
      -- Fully-connected: 4 x 4 x z --> 4 x 4 x baseChannels
      decoder:add(nn.Linear(z*w*w, w*w*baseChannels))
      decoder:add(nn.View(baseChannels, w, w):setNumInputDims(1))

      -- deconv4-1, deconv4-2: 4 x 4 --> 8 x 8
      decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      baseChannels = baseChannels/2

      -- deconv3: 8 x 8 --> 16 x 16
      decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      baseChannels = baseChannels/2

      -- deconv2: 16 x 16 --> 32 x 32
      decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))

      -- deconv1: 32 x 32 --> 64 x 64
      decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))

      -- tanH: 64 x 64 --> 64 x 64
      decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.SpatialBatchNormalization(baseChannels, eps, mom))
      decoder:add(nn.LeakyReLU(0.1))
      decoder:add(cudnn.SpatialConvolution(baseChannels, 3, 3, 3, 1, 1, 1, 1))
      decoder:add(nn.Tanh())

      -------------------
      -- Discriminator --
      -------------------
      -- conv1: 64 x 64 --> 32 x 32
      baseChannels = baseChannels/2
      gan:add(cudnn.SpatialConvolution(3, baseChannels, 5, 5, 1, 1, 2, 2))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- conv2: 32 x 32 --> 16 x 16
      gan:add(cudnn.SpatialConvolution(baseChannels, baseChannels*2, 5, 5, 1, 1, 2, 2))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- conv3: 16 x 16 --> 8 x 8
      gan:add(cudnn.SpatialConvolution(baseChannels*2, baseChannels*4, 5, 5, 1, 1, 2, 2))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- conv4: 8 x 8 --> 4 x 4
      gan:add(cudnn.SpatialConvolution(baseChannels*4, baseChannels*8, 3, 3, 1, 1, 1, 1))
      gan:add(cudnn.SpatialMaxPooling(2, 2))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.SpatialDropout(0.5, true))

      -- Fully-connected: 4 x 4 x (baseChannels x 8) --> 4 x 4 x (baseChannels x 2)
      -- followed by batch discrimination
      gan:add(nn.View(w*w*baseChannels*8))
      gan:add(nn.Linear(w*w*baseChannels*8, w*w*baseChannels*2))
      gan:add(nn.Normalize(2))
      gan:add(nn.Dropout(0.5, true))
      gan:add(cudnn.ReLU(true))
      gan:add(nn.BatchDiscrimination(w*w*baseChannels*2, 100, 5))
      gan:add(nn.Linear(w*w*baseChannels*2 + 100, 1))
      gan:add(nn.Sigmoid())
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k, v in pairs(encoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
      for k, v in pairs(decoder:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:uniform(-1*math.sqrt(1/n), math.sqrt(1/n))
         if not opt.bias then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k, v in pairs(encoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
      for k, v in pairs(decoder:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('cudnn.SpatialFullConvolution')
   ConvInit('nn.SpatialFullConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')

   for k, v in pairs(encoder:findModules('nn.Linear')) do
      v.bias:zero()
   end
   for k, v in pairs(decoder:findModules('nn.Linear')) do
      v.bias:zero()
   end

   if opt.cudnn == 'deterministic' then
      encoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
      decoder:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   sampling_z = nn.Sampler()
   KLD = nn.KLDCriterion()
   BCECriterion = nn.BCECriterion()
   ReconCriterion = nn.MSECriterion()

   encoder:cuda()
   decoder:cuda()
   gan:cuda()
   encoder:cuda()
   decoder:cuda()
   ReconCriterion:cuda()
   KLD:cuda()
   sampling_z:cuda()
   BCECriterion:cuda()
   gan:cuda()

   return {encoder, decoder, gan, sampling_z}, {KLD, ReconCriterion, BCECriterion}
end

return createModel
