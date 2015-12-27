local ffi=require 'ffi'

function makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      cutorch.setDevice(opt.GPU)
   end
   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU)
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function rand_initialize(layer)
  local tn = torch.type(layer)
  if tn == "cudnn.SpatialConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "cudnn.VolumetricConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.Linear" then
    local c =  math.sqrt(2.0 / layer.weight:size(2));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
--  elseif layer.weight or layer.bias then
--    -- If there is any parameterized layer that skips the new initialization,
--    -- fail early so that users could notice.
--    error("Layer is trainable but not initialized!");
  end
end
