function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(cudnn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = cudnn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            features:add(conv3)
            --features:add(nn.SpatialBatchNormalization(oChannels))
            features:add(cudnn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end
   features:add(cudnn.SpatialConvolution(512,4096,7,7))
   --features:add(cudnn.ReLU(true))
   features:add(nn.SpatialBatchNormalization(4096))
   features:add(cudnn.ReLU(true))
   
   features:add(nn.SpatialDropout(0.5))
   features:add(cudnn.SpatialConvolution(4096,4096,1,1))
   --features:add(cudnn.ReLU(true))
   features:add(nn.SpatialBatchNormalization(4096))
   features:add(cudnn.ReLU(true))
 
   features:add(nn.SpatialDropout(0.5))

   features:cuda()
   features = makeDataParallel(features, nGPU) -- defined in util.lua

   

   local classifier = nn.Sequential()
   classifier:add(cudnn.SpatialConvolution(4096, nClasses,1,1))
   classifier:add(nn.View(nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:cuda()

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end
