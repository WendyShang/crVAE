local function ConcatAct()
    local model = nn.ConcatTable()
    model:add(nn.Identity())
    model:add(nn.MulConstant(-1))
    return model
end

local function createModel(opt)
    local encoder = nn.Sequential()
    local decoder = nn.Sequential()
    local baseChannels = opt.baseChannels
    local z = opt.nf
    local w = opt.latentDims[1]
    local eps = opt.eps
    local mom= opt.mom

    if opt.dataset == 'mnist_28x28' then
        ---------------------------------
        -- Encoder (Inference network) --
        -- convolution net -> FC layer --
        ---------------------------------
        --conv1-1, conv1-2: 28x28 --> 16 x 16
        encoder:add(cudnn.SpatialConvolution(1, baseChannels, 5, 5, 2, 2, 3, 3))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        encoder:add(ConcatAct())
        encoder:add(nn.JoinTable(2))
        encoder:add(cudnn.ReLU(true))
        baseChannels = baseChannels * 2
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        encoder:add(nn.LeakyReLU(0.1))

        --conv2-1, conv2-2: 16x16 --> 8 x 8
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 2, 2, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels, 1e-6,0.9))
        encoder:add(ConcatAct())
        encoder:add(nn.JoinTable(2))
        encoder:add(cudnn.ReLU(true))
        baseChannels = baseChannels * 2
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        encoder:add(nn.LeakyReLU(0.1))

        --conv3-1, conv3-2: 8 x 8 --> 4 x4 -- 8 x 8 --> 4 x 4
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 5, 5, 2, 2, 2, 2))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        encoder:add(ConcatAct())
        encoder:add(nn.JoinTable(2))
        encoder:add(cudnn.ReLU(true))
        baseChannels = baseChannels * 2
        encoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        encoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        encoder:add(nn.LeakyReLU(0.1))

        -- variance path: FC followed 
        -- mean path: FC followed 
        encoder:add(nn.View(w*w*baseChannels):setNumInputDims(3))
        local mean_logvar = nn.ConcatTable()
        mean_logvar:add(nn.Linear(w*w*baseChannels, w*w*z))
        mean_logvar:add(nn.Linear(w*w*baseChannels, w*w*z))
        encoder:add(mean_logvar)

        -- deconv5: 4 x 4 --> 4 x 4
        decoder:add(nn.Linear(z*w*w, w*w*baseChannels))
        decoder:add(nn.View(baseChannels, w, w):setNumInputDims(1))

        -- deconv4-1, conv4-2: 4 x 4 --> 8 x 8
        decoder:add(cudnn.SpatialFullConvolution(baseChannels, baseChannels, 4, 4, 2, 2, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        decoder:add(nn.LeakyReLU(0.1))

        -- deconv3-1, conv3-2: 8 x 8 --> 16 x 16
        baseChannels = baseChannels/2
        decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        decoder:add(nn.LeakyReLU(0.1))

        -- deconv3-1, conv3-2, conv3-3: 16 x 16 --> 28 x 28
        baseChannels = baseChannels/2
        decoder:add(cudnn.SpatialFullConvolution(baseChannels*2, baseChannels, 4, 4, 2, 2, 2, 2))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
        decoder:add(nn.LeakyReLU(0.1))
        decoder:add(cudnn.SpatialConvolution(baseChannels, baseChannels, 3, 3, 1, 1, 1, 1))
        decoder:add(nn.SpatialBatchNormalization(baseChannels,1e-6,0.9))
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

    sampling_z = nn.Sampler()
    ReconCriterion = nn.BCECriterion()
    KLD = nn.KLDCriterion()
    
    KLD:cuda()
    encoder:cuda()
    decoder:cuda()
    sampling_z:cuda()
    ReconCriterion:cuda()

    return {encoder, decoder, sampling_z}, {KLD, ReconCriterion}
end

return createModel
