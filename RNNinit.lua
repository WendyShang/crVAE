ffi = require 'ffi'
function firstToUpper(str)
    return str:gsub("^%l", string.upper)
end

-- | Get the parameters for a linear projection.
-- rnn     : The Cudnn module.
-- layer   : The layer number (0 indexed).
-- layerId : The projection in the layer (0 indexed).
function getParams(rnn, layer, layerId)
    if not rnn.wDesc then
        rnn:resetWeightDescriptor()
    end
    local fns = {
        weight = 'cudnnGetRNNLinLayerMatrixParams',
        bias   = 'cudnnGetRNNLinLayerBiasParams',
    }
    local params = {}
    for key, fn in pairs(fns) do
        local desc = rnn:createFilterDescriptors(1)
        local pointer = ffi.new("float*[1]")
        cudnn.errcheck(
            fn,
            cudnn.getHandle(),
            rnn.rnnDesc[0],
            layer,
            rnn.xDescs[0],
            rnn.wDesc[0],
            rnn.weight:data(),
            layerId,
            desc[0],
            ffi.cast("void**", pointer)
        )
        local dataType = ffi.new("cudnnDataType_t[1]")
        local format   = ffi.new("cudnnTensorFormat_t[1]")
        local nbDims   = torch.IntTensor(1)
        local minDim = 3
        local filterDimA = torch.ones(minDim):int()
        cudnn.errcheck(
            'cudnnGetFilterNdDescriptor',
            desc[0],
            minDim,
            dataType,
            format,
            nbDims:data(),
            filterDimA:data()
        )
        local offset = pointer[0] - rnn.weight:data()
        params[key] = torch.CudaTensor(
            rnn.weight:storage(), offset + 1, filterDimA:prod())
        params["grad" .. firstToUpper(key)] = torch.CudaTensor(
            rnn.gradWeight:storage(), offset + 1, filterDimA:prod())
    end
    return params
end
