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
    local baseChannels = opt.baseChannels
    local w = opt.latentDims[1]
    local z = opt.nf
    local eps = opt.eps
    local mom = opt.mom
    local time_step = opt.timeStep

    if opt.dataset == 'mnist_28x28' then
        -----------------------------------
        -- Encoder (Inference network) ----
        -- convolution net -> LSTM layer --
        -----------------------------------
        --conv1-1, conv1-2: 28x28 --> 16 x 16
        encoder:add(cudnn.SpatialConvolution(1, baseChannels, 5, 5, 2, 2, 3, 3))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        encoder:add(ConcatAct())
        encoder:add(nn.JoinTable(2))
        encoder:add(cudnn.ReLU(true))
        baseChannels = baseChannels * 2
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        encoder:add(nn.LeakyReLU(0.1))

        --conv2-1, conv2-2: 16x16 --> 8 x 8
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels, eps,mom))
        encoder:add(ConcatAct())
        encoder:add(nn.JoinTable(2))
        encoder:add(cudnn.ReLU(true))
        baseChannels = baseChannels * 2
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        encoder:add(nn.LeakyReLU(0.1))

        --conv3-1, conv3-2: 8 x 8 --> 4 x4 
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels, eps,mom))
        encoder:add(ConcatAct())
        encoder:add(nn.JoinTable(2))
        encoder:add(cudnn.ReLU(true))
        baseChannels = baseChannels * 2
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        encoder:add(nn.LeakyReLU(0.1))

        -- LSTM layer for Channel-Recurrency
        -- mean path: FC followed by bias subtraction 
        local mean_shift = nn.Sequential()
        local add_size = torch.LongStorage(3)
        add_size[1] = z
        add_size[2] = w
        add_size[3] = w
        mean_shift:add(nn.View(w*w*baseChannels):setNumInputDims(3))
        mean_shift:add(nn.Linear(w*w*baseChannels, w*w*z))
        mean_shift:add(nn.View(z, w, w):setNumInputDims(1))
        mean_shift:add(nn.Add(add_size))
        mean_shift:get(4):reset(0.01)

        local mean_logvar_before = nn.ConcatTable()
        mean_logvar_before:add(mean_shift)
        mean_logvar_before:add(nn.Identity())
        encoder:add(mean_logvar_before)

        -- variance path: 
        -- 1. 4 x 4 x baseChannels is divided into time_step blocks of size 4 x 4 x baseChannels/time_step
        -- 2. channel-recurrency via LSTM followed by block-wise FC layer to generate \sigma^{rnn} 
        local input_division = baseChannels/time_step
        local latent_division = z/time_step
        local var_transform = nn.Sequential()
        local LSTM_encoder = cudnn.LSTM(w*w*input_division, w*w*input_division, opt.rnnLayer, true)
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
        local LSTM_decoder = cudnn.LSTM(w*w*latent_division, w*w*input_division, opt.rnnLayer, true)
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

        -- add bias back
        local add_size = torch.LongStorage(3)
        add_size[1] = z
        add_size[2] = w
        add_size[3] = w
        decoder:add(nn.Add(add_size))
        decoder:get(1):reset(0.01)

        -- deconv5: 4 x 4 --> 4 x 4
        decoder:add(cudnn.SpatialConvolution(z, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))

        -- deconv4-1, conv4-2: 4 x 4 --> 8 x 8
        decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        decoder:add(nn.LeakyReLU(0.1))
        baseChannels = baseChannels/2

        -- deconv3-1, conv3-2: 8 x 8 --> 16 x 16
        decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        decoder:add(nn.LeakyReLU(0.1))
        baseChannels = baseChannels/2

        -- deconv3-1, conv3-2, conv3-3: 16 x 16 --> 28 x 28
        decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 2, 2))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,eps,mom))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, 1, 3, 3, 1, 1, 0, 0))
        decoder:add(nn.Sigmoid())
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
    BNInit('nn.SpatialBatchNormalization')

    for k,v in pairs(encoder:findModules('nn.Linear')) do
        v.bias:zero()
    end
    for k,v in pairs(decoder:findModules('nn.Linear')) do
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


    ReconCriterion = nn.BCECriterion()
    KLD = nn.KLDCriterion()
    sampling_z = nn.Sampler()

    encoder:cuda()
    decoder:cuda()
    var_encoder:cuda()
    var_decoder:cuda()
    sampling_z:cuda()
    ReconCriterion:cuda()
    KLD:cuda()

    return {encoder, decoder, var_encoder, var_decoder, sampling_z}, {KLD, ReconCriterion}
end

return createModel
