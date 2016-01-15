function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32
   local bottleneck_flag =0;
   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   -- cfg = {nElems, nChannels, Stride}
   if modelType == 'A' then
      cfg = {{3,256,1},{4,512,2},{6,1024,2},{3,2048,2}}
   elseif modelType == 'B' then
      cfg = {{3,64,1},{4,128,2},{36,256,2},{3,512,2}}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   if bottleneck_flag == 0 then
      for k,v in ipairs(cfg) do
         v[2]=v[2]/4;
      end
   end

   
   function createResElem(iChannels,oChannels,stride) 
      local bottleneck = nn.Sequential()
      if bottleneck_flag == 1 then 
         bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels/4,1,1,stride,stride,0,0))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels/4))
         bottleneck:add(cudnn.ReLU(true))
         bottleneck:add(cudnn.SpatialConvolution(iChannels/4,iChannels/4,3,3,1,1,1,1))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels/4))
         bottleneck:add(cudnn.ReLU(true))
         bottleneck:add(cudnn.SpatialConvolution(iChannels/4,oChannels,1,1,1,1,0,0))
         bottleneck:add(cudnn.SpatialBatchNormalization(oChannels))
      else
         bottleneck:add(cudnn.SpatialConvolution(iChannels,oChannels,3,3,stride,stride,1,1))
         bottleneck:add(cudnn.SpatialBatchNormalization(oChannels))
         --bottleneck:add(cudnn.ReLU(true))
         bottleneck:add(cudnn.SpatialConvolution(oChannels,oChannels,3,3,1,1,1,1))
         bottleneck:add(cudnn.SpatialBatchNormalization(oChannels))
         --bottleneck:add(cudnn.ReLU(true))
      end 


      local shortcut = nn.Sequential();
      if stride > 1 then
         --local bconv = nn.Sequential();
         --bconv:add(cudnn.SpatialConvolution(iChannels,oChannels-iChannels,1,1,stride,stride,0,0))
         --bconv:add(nn.SpatialBatchNormalization(oChannels-iChannels))
         --local con = nn.DepthConcat(2);
         --con:add(cudnn.SpatialAveragePooling(stride,stride))
         --con:add(cudnn.SpatialConvolution(iChannels,oChannels-iChannels,1,1,stride,stride,0,0))
         --shortcut:add(con)
         shortcut:add(cudnn.SpatialConvolution(iChannels,oChannels,1,1,stride,stride,0,0))
         shortcut:add(cudnn.SpatialBatchNormalization(oChannels))
      else 
         shortcut:add(nn.Identity())
      end
      

      local hob = nn.ConcatTable()
      hob:add(bottleneck):add(shortcut)
      
      local resElem = nn.Sequential()
      resElem:add(hob):add(nn.CMaxTable(true))--:add(cudnn.ReLU(true))
      resElem:add(cudnn.SpatialBatchNormalization(oChannels))
      
      return resElem
   end

   local features=nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,cfg[1][2],7,7,2,2,3,3))
   features:add(cudnn.SpatialBatchNormalization(cfg[1][2]))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(2,2))
   
   
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
  

   features:add(cudnn.SpatialConvolution(iChannels,nClasses,1,1))   
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
   features:apply(rand_initialize)
   
   features:cuda()   
   features = makeDataParallel(features, nGPU) -- defined in util.lua
 


   local model = features
   
   return model
end
