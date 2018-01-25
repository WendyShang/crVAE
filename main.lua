--Import Libraries
package.path = package.path .. ';?/?.lua;./?.lua' 
require 'torch';
require 'nn';
require 'cunn';
require 'optim';
require 'image';
require 'stn';
require 'pl';
require 'paths';
require 'cutorch';
require 'cudnn';

require 'Sampler';
require 'KLDCriterion';
require 'RNNinit';

init = require 'init';
opts = require 'opts';
image_utils = require 'image_utils';

--House Keeping
opt = opts.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

if opt.alpha1 and opt.alpha2 then
    opt.save = opt.save  .. opt.latentType  .. '_LR_' .. opt.LR .. '_alpha_' .. opt.alpha1 .. '_' .. opt.alpha2 .. '_beta_' .. opt.beta .. '_ka' .. opt.kappa .. '/'
else
    opt.save = opt.save  .. opt.latentType  .. '_LR_' .. opt.LR .. '_alpha_' .. opt.alpha .. '_beta_' .. opt.beta ..  '/'
end

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

init.savetable(opt, opt.save .. 'hyperparameters.txt')

--Data Loading
data = {}
data.train = torch.load(opt.data .. '/train_bird.t7')
data.train = data.train:mul(2):add(-1)
data.test = torch.load(opt.data .. '/test_bird.t7')
data.test = data.test:mul(2):add(-1)
opt.scales = {64}
opt.latentDims = {4}
opt.latentFilters = {opt.nf}
opt.preprocess_train = image_utils.HorizontalFlip(0.5)
opt.preprocess_test = image_utils.HorizontalFlip(1)

--Import Model, Criterion and Training functions
local createModel = require('models/' .. opt.netType .. '_' .. opt.latentType)
models, criterions = createModel(opt)
require('train_test/train_test_' .. opt.latentType)

--Start Training
epoch = opt.epochNumber
for i = opt.epochNumber, opt.nEpochs do
   train(opt)
   collectgarbage()
   val(opt)
   collectgarbage()
   epoch = epoch + 1
end