--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   RP,
        {  1,     10,   1e-1,   1 },
        { 11,     15,    1e-2,   1  },
        { 16,     20 ,  1e-3,   1 },
        { 21,    30,   1e-4,   1 },
    }
    for _, row in ipairs(regimes) do
       if epoch >= row[1] and epoch <= row[2] then
          return { learningRate=row[3], weightDecay=1e-4, repeatBatch=row[4] }, epoch == row[1]
       end
    end
--     local c= 5e-4;
--     local A = 0.112;
--     local T = 7;
--     local t =epoch
--     local u = A*(1+(c/A)*(t/T))/(1+(c/A)*((t^4)/T)+T*(t^2)/(T^2))
-- return {learningRate = u, weightDecay=0}, epoch ==epoch
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   opt.testFlag = 0; 
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay,
         repeatBatch = params.repeatBatch
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   local function sanitize(net)
      local list = net:listModules()
      for _,val in ipairs(list) do
            for name,field in pairs(val) do
               if torch.type(field) == 'cdata' then val[name] = nil end
               if (name == 'output' or name == 'gradInput') then
                  if torch.type(field) == 'table' then
                     for i,f in ipairs(field) do
                      if torch.type(f) ~= 'table'  then 
                        val[name][i] = f.new()
                      end
                    end
                  else
                    val[name] = field.new()
                end
               end
            end
      end
   end
   sanitize(model)
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()
local convNodes = model:findModules('cudnn.SpatialConvolution')

function model:BinaryForward(X)
  local realParams = parameters:clone()
  for i =1, #convNodes do
     if i ~= #convNodes then
       cutorch.setDevice(math.ceil(i/(#convNodes/opt.nGPU)))
     else
      cutorch.setDevice(opt.GPU)
     end

     local n = convNodes[i].weight[1]:nElement()
     local s = convNodes[i].weight:size()
     local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n);
     --m=m:expand(s)
     convNodes[i].weight:sign():cmul(m:expand(s))
     --convNodes[i].bias:add(1):div(2):cmin(1):cmax(-1e-6):sign()
   end
   cutorch.setDevice(opt.GPU)
  f = model:forward(X)
  parameters:copy(realParams);
  return f
end

function model:BinaryBackward(X,L)
  local realParams = parameters:clone()
  for i =1, #convNodes do
     if i ~= #convNodes then
       cutorch.setDevice(math.ceil(i/(#convNodes/opt.nGPU)))
     else
      cutorch.setDevice(opt.GPU)
     end

     local n = convNodes[i].weight[1]:nElement()
     local s = convNodes[i].weight:size()
     local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n);
     --m=m:expand(s)--repeatTensor(1,s[2],s[3],s[4])
     convNodes[i].weight:sign():cmul(m:expand(s))

     --convNodes[i].bias:add(1):div(2):cmin(1):cmax(-1e-6):sign()
   end
   cutorch.setDevice(opt.GPU)
  b = model:backward(X,L)
  parameters:copy(realParams);
  return b
end

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs
   if opt.binaryWeight == 1 then
     feval = function(x)
       model:zeroGradParameters()
         local realParams = parameters:clone()
         for i =1, #convNodes do
           --if i ~= #convNodes then
             cutorch.setDevice(math.ceil(i/(#convNodes/opt.nGPU)))
           --else
           --  cutorch.setDevice(opt.GPU)
           --end

           local n = convNodes[i].weight[1]:nElement()
           local s = convNodes[i].weight:size()
           local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n);
           convNodes[i].weight:sign():cmul(m:expand(s))
         end
         cutorch.setDevice(opt.GPU)
         outputs = model:forward(inputs)
         err = criterion:forward(outputs, labels)
         local gradOutputs = criterion:backward(outputs, labels)
         model:backward(inputs, gradOutputs)
         parameters:copy(realParams);
         return err, gradParameters
     end
   else
     feval = function(x)
       model:zeroGradParameters()
       outputs = model:forward(inputs)
       err = criterion:forward(outputs, labels)
       local gradOutputs = criterion:backward(outputs, labels)
       model:backward(inputs, gradOutputs)
       return err, gradParameters
     end
   end 

   for i = 1,optimState.repeatBatch do
     optim.sgd(feval, parameters, optimState)
   end
   -- DataParallelTable's syncParameters
   model:apply(function(m) if m.syncParameters then m:syncParameters() end end)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labelsCPU[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
      end
      top1 = top1 * 100 / opt.batchSize;
   end
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end
