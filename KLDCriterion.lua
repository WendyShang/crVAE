local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

function KLDCriterion:updateOutput(mean, log_var)
    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()
    KLDelements:exp():mul(-1)
    KLDelements:add(-1, mean_sq)
    KLDelements:add(1)
    KLDelements:add(log_var)

    local output = -0.5 * torch.sum(KLDelements)
    if self.sizeAverage then
        output = output / mean:size(1)
    end
    self.output = output
    return self.output
end

function KLDCriterion:updateGradInput(mean, log_var)
	self.gradInput = {}

    self.gradInput[1] = mean:clone()

    self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5)
    if self.sizeAverage then
        self.gradInput[1] = self.gradInput[1]/(mean:size(1))
        self.gradInput[2] = self.gradInput[2]/(log_var:size(1))
    end
    return self.gradInput
end
