function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   -- cfg = {nElems, nChannels, Stride}
   if modelType == 'A' then
      --cfg = {{4,64},{'M',2},{'D',128},{3,128},{'M',2},{'D',256},{4,256},{'M',2},{'D',512},{3,512},{'D',512}}
      cfg = {{3,64,128},{'M',2},{4,128,256},{'M',2},{4,256,512},{'M',2},{4,512,512}}
   elseif modelType == 'B' then
      cfg = {{2,64},{'M',2},{'D',128},{3,128},{'M',2},{'D',256},{6,256},{'M',2},{'D',256},{2,256}}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end
  
   function createElem(iChannels,oChannels, nNodes) 
      local upElem = nn.Sequential()
      for i=1,nNodes do   
         local bottleneck = nn.Sequential()
         bottleneck:add(nn.SelectTable(1))        
         bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels,1,1,1,1,0,0))
         bottleneck:add(cudnn.ReLU(true))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels))
         bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels,3,3,1,1,1,1))
         bottleneck:add(cudnn.ReLU(true))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels))
         
         --bottleneck:add(cudnn.SpatialConvolution(iChannels/4,oChannels,1,1,1,1,0,0))
         --bottleneck:add(cudnn.ReLU(true))
         --bottleneck:add(nn.SpatialBatchNormalization(oChannels))
         
         local hob = nn.ConcatTable():add(bottleneck):add(nn.SelectTable(1));
         local sumElem = nn.Sequential():add(hob):add(nn.CMaxTable(true))
         upElem:add(nn.ConcatTable():add(sumElem):add(nn.Identity()))
      end
      upElem:add(nn.FlattenTable()):add(nn.JoinTable(2))
      upElem:add(cudnn.SpatialConvolution(iChannels*(nNodes+1),oChannels,1,1,1,1,0,0)) 
      
      --upElem:add(nn.FlattenTable()):add(nn.CAddTable())
      --upElem:add(cudnn.SpatialConvolution(iChannels,oChannels,1,1,1,1,0,0)) 

      upElem:add(cudnn.SpatialBatchNormalization(oChannels))           
      upElem:add(nn.ConcatTable():add(nn.Identity()))
      
      return upElem
   end

   local features=nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,cfg[1][2],7,7,2,2,3,3));
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialBatchNormalization(cfg[1][2]));

   
   features:add(cudnn.SpatialMaxPooling(2,2))
   
   features:add(nn.ConcatTable():add(nn.Identity()))
   --features:apply(rand_initialize)
   
      --local  iChannels = 3 --cfg[1][2];
      local  iChannels = cfg[1][2]; 
      for k,v in ipairs(cfg) do
         if v[1]=='M' then 
            features:add(nn.SelectTable(1))
            features:add(cudnn.SpatialMaxPooling(v[2],v[2]))
            features:add(nn.ConcatTable():add(nn.Identity()))
         elseif v[1] == 'D' then
            features:add(nn.SelectTable(1))
            features:add(cudnn.SpatialConvolution(oChannels,v[2],1,1,1,1,0,0))  
            --features:add(nn.SpatialCMul(v[2]))
            features:add(cudnn.ReLU(true))          
            features:add(cudnn.SpatialBatchNormalization(v[2]))
            
            features:add(nn.ConcatTable():add(nn.Identity()))
            oChannels = v[2];
           else
           
              local nNodes = v[1]
              oChannels = v[3]
              iChannels = v[2]
              local resElem = createElem(iChannels,oChannels,nNodes)
              features:add(resElem)
              --iChannels = oChannels;
                         
         end
      end
   features:add(nn.SelectTable(1)):add(cudnn.SpatialAveragePooling(7,7))

  
   
   
  
   features:add(cudnn.SpatialConvolution(oChannels,nClasses,1,1))
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
   --features:apply(rand_initialize)
   --features:cuda()

   features:apply(rand_initialize)
   features:cuda()   
   features = makeDataParallel(features, nGPU) -- defined in util.lua


   local model = features -- nn.Sequential()
   --model:add(features):add(classifier)

   return model
end
