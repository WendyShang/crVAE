function optim.adam_gan(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8
    local step = config.step or 1

    if config.numUpdates >= 30 and config.numUpdates % step ~= 0 then
        config.optimize = false
    end

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
      -- Initialization
      state.t = state.t or 0
      -- Exponential moving average of gradient values
      state.m = state.m or x.new(dfdx:size()):zero()
      -- Exponential moving average of squared gradient values
      state.v = state.v or x.new(dfdx:size()):zero()
      -- A tmp tensor to hold the sqrt(v) + epsilon
      state.denom = state.denom or x.new(dfdx:size()):zero()

      state.t = state.t + 1

      -- Decay the first and second moment running average coefficient
      state.m:mul(beta1):add(1-beta1, dfdx)
      state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

      state.denom:copy(state.v):sqrt():add(epsilon)

      local biasCorrection1 = 1 - beta1^state.t
      local biasCorrection2 = 1 - beta2^state.t

    local fac = 1
    if config.numUpdates < 10 then
        fac = 50.0
    elseif config.numUpdates < 30 then
        fac = 5.0
    else
        fac = 1.0
    end

    io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
      -- (2) update x
      x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end
