--Import Libraries
package.path = package.path .. ';?/?.lua;./?.lua' -- FB
require 'torch';
require 'nn';
require 'cunn';
require 'optim';
require 'image';
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

local function loadBinarizedMNIST(binaryMNIST)
  print ('Loading Binarized MNIST dataset')
  local train = torch.load(binaryMNIST .. '/binarized_mnist_train.t7')
  local test  = torch.load(binaryMNIST .. '/binarized_mnist_test.t7')
  local valid = torch.load(binaryMNIST .. '/binarized_mnist_valid.t7')
  local data = {}
  data.train = torch.cat(train,valid,1)
  data.test  = test
  data.dim_input = 784
  collectgarbage()
  return data
end

--House Keeping
opt = opts.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

opt.save = opt.save  .. 'dataset_' .. opt.dataset .. '_' .. opt.netType .. '_' .. opt.latentType .. '_optim_' .. opt.optimization .. '_nf_' .. opt.nf .. '_ts_' .. opt.timeStep .. '_LR_' .. opt.LR .. '_alpha_' .. opt.alpha .. '/'

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

init.savetable(opt, opt.save .. 'hyperparameters.txt')

--Data Loading
data = {}
--test with binary MNIST
data = loadBinarizedMNIST(opt.binaryMNIST)
data.test = data.test:resize(10000, 1, 28, 28):float()
--train with dynamic MNIST
local mnist_data = torch.load(opt.dynamicMNIST .. '/mnist_train.t7')
data.train = mnist_data.digits:float():mul(1/255)
opt.scales = {28}
opt.latentDims = {4}
opt.latentFilters = {opt.nf}

--Import Model, Criterion and Training functions
local createModel = require('models/' .. opt.netType .. '_' .. opt.latentType .. '_mnist')
models, criterions = createModel(opt)
require('train_test/train_test_' .. opt.latentType .. '_mnist')

--Training and evaluation
best_vlb = -1000
epoch = opt.epochNumber
for i = opt.epochNumber, opt.nEpochs do
    train(opt)
    collectgarbage()
    vlb = val(opt)
    collectgarbage()
    epoch = epoch + 1
    if best_vlb < vlb then
        torch.save(opt.save .. '/model_best.t7', {vae_encoder:clearState(), vae_decoder:clearState()})
        best_vlb = vlb
    end
    print('Saving dir: ' .. opt.save)
end
print('Best vlb is: '.. best_vlb)
print('Testing final nll and vlb...Please be patient!')
final_nll_100, final_vlb_100 = evaluate(opt, 100)
print('The 100 sample nll is ' .. final_nll_100 .. ' and vlb ' .. final_vlb_100)
final_nll, final_vlb = evaluate(opt, 10000)
print('The 10K sample nll is ' .. final_nll .. ' and vlb ' .. final_vlb)
