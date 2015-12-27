function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   -- cfg = {nElems, nChannels, Stride}
   if modelType == 'A' then
      cfg = {{3,256,1},{4,512,2},{6,1024,2},{3,2048,2}}
   elseif modelType == 'B' then
      cfg = {{3,64,1},{4,128,2},{6,256,2},{3,512,2}}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end
   
   function createResElem(iChannels,oChannels,stride) 
      local bottleneck = nn.Sequential()
      bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels/4,1,1,stride,stride,0,0))
      --bottleneck:add(nn.SpatialBatchNormalization(iChannels/4))
      bottleneck:add(cudnn.ReLU(true))
      bottleneck:add(cudnn.SpatialConvolution(iChannels/4,iChannels/4,3,3,1,1,1,1))
      bottleneck:add(nn.SpatialBatchNormalization(iChannels/4))
      bottleneck:add(cudnn.ReLU(true))
      bottleneck:add(cudnn.SpatialConvolution(iChannels/4,oChannels,1,1,1,1,0,0))
      --bottleneck:add(nn.SpatialBatchNormalization(oChannels))
      --bottleneck:apply(rand_initialize)

      local shortcut = nn.Sequential();
      if stride > 1 then
         local con = nn.DepthConcat(2);
         con:add(cudnn.SpatialAveragePooling(stride,stride))
         con:add(cudnn.SpatialConvolution(iChannels,oChannels-iChannels,1,1,stride,stride,0,0))
         --con:get(2).weight:fill(1);
         --con:get(2).bias:fill(0);
         shortcut:add(con)
         --shortcut:get(1).weight:fill(1);
         --shortcut:get(1).bias:fill(0);
         --shortcut:add(nn.SpatialBatchNormalization(oChannels,1e-3))
         --shortcut:add(cudnn.SpatialMaxPooling(stride,stride))
      else 
         shortcut:add(nn.Identity())
      end
      

      local hob = nn.ConcatTable()
      hob:add(bottleneck):add(shortcut)
      
      local resElem = nn.Sequential()
      resElem:add(hob):add(nn.CAddTable()):add(cudnn.ReLU(true))
      
      return resElem
   end

   local features=nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,cfg[1][2],7,7,2,2,3,3))
   features:add(nn.SpatialBatchNormalization(cfg[1][2]))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(2,2))
   --features:apply(rand_initialize)
   
      local  iChannels = cfg[1][2];
      local  oChannels = 0; 
      for k,v in ipairs(cfg) do
          oChannels = v[2];
         for i = 1,v[1] do
            local stride = ((i==1) and v[3]) or 1;
            local resElem = createResElem(iChannels,oChannels,stride)
            features:add(resElem)
            iChannels = oChannels;
         end
      end
   features:add(cudnn.SpatialAveragePooling(7,7))

   features:cuda()   
   features = makeDataParallel(features, nGPU) -- defined in util.lua
   
   
   local classifier = nn.Sequential()
   classifier:add(cudnn.SpatialConvolution(iChannels,nClasses,1,1))
   --classifier:apply(rand_initialize)
   classifier:add(nn.View(nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()


   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end
