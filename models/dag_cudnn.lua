function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   -- cfg = {nElems, nChannels, Stride}
   if modelType == 'A' then
      cfg = {{4,64},{'M',2},{'D',128},{3,128},{'M',2},{'D',256},{4,256},{'M',2},{'D',512},{3,512},{'D',512}}
   elseif modelType == 'B' then
      cfg = {{3,64},{'M',2},{'D',128},{4,128},{'M',2},{'D',256},{15,256},{'M',2},{'D',512},{3,512},{'D',512}}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end
   
   function createElem(iChannels,oChannels, nNodes) 
      local upElem = nn.Sequential()
      for i=1,nNodes do
         upElem:add(nn.ConcatTable())
         for j=1,i do
            upElem:get(i):add(nn.SelectTable(j))
            if j==i then
               local conv = nn.Sequential();
                     conv:add(cudnn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1));
                     --conv:add(nn.SpatialCMul(oChannels))
                     conv:add(cudnn.ReLU(true));
                     conv:add(nn.SpatialBatchNormalization(oChannels));
                     --conv:add(cudnn.ReLU());

               local convS = nn.Sequential():add(nn.SelectTable(i)):add(conv);      
               local sumT = nn.ConcatTable()
               for k=1,j do
                  if k==j then
                     sumT:add(convS);
                  else
                     sumT:add(nn.SelectTable(k));
                  end
               end
               upElem:get(i):add(nn.Sequential():add(sumT):add(nn.CAddTable()):add(nn.MulConstant(1/j,true)));
            end 
         end
      end
      upElem:add(nn.FlattenTable()):add(nn.JoinTable(2)):add(nn.ConcatTable():add(nn.Identity()))
      return upElem
   end

   local features=nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,cfg[1][2],7,7,2,2,3,3));
   --features:add(nn.SpatialCMul(cfg[1][2]))
   features:add(cudnn.ReLU(true))
   features:add(nn.SpatialBatchNormalization(cfg[1][2]));
   --features:add(cudnn.ReLU())
   
   features:add(cudnn.SpatialMaxPooling(2,2))
   features:add(nn.ConcatTable():add(nn.Identity()))
   --features:apply(rand_initialize)
   
      --local  iChannels = 3 --cfg[1][2];
      local  oChannels = cfg[1][2]; 
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
            features:add(nn.SpatialBatchNormalization(v[2]))
            
            features:add(nn.ConcatTable():add(nn.Identity()))
            oChannels = v[2];
         else
           
              local nNodes = v[1]
              local resElem = createElem(oChannels,oChannels,nNodes)
              features:add(resElem)
            oChannels = oChannels*(nNodes+1)             
         end
      end
   features:add(nn.SelectTable(1)):add(cudnn.SpatialAveragePooling(7,7))

   features:apply(rand_initialize)
   features:cuda()   
   features = makeDataParallel(features, nGPU) -- defined in util.lua
   
   
   local classifier = nn.Sequential()
   classifier:add(cudnn.SpatialConvolution(oChannels,nClasses,1,1))
   classifier:add(nn.View(nClasses))
   classifier:add(nn.LogSoftMax())
   classifier:apply(rand_initialize)
   classifier:cuda()


   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end
