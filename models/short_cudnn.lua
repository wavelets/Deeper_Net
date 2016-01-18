function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   -- cfg = {nElems, nChannels, Stride}
   if modelType == 'A' then
      --cfg = {{3,64,128,1},{4,128,256,2},{6,256,512,2},{3,512,512,2}}
      --cfg = {{2,64,128,1},{'M',2},{2,128,256,1},{'M',2},{2,256,512,1},{'M',2},{2,512,512,1}}
      cfg = {{3,4*64,4*128,1},{'M',2},{4,4*128,4*256,1},{'M',2},{6,4*256,4*512,1},{'M',2},{3,4*512,4*512,1}}
   elseif modelType == 'B' then
      cfg = {{2,64},{'M',2},{'D',128},{3,128},{'M',2},{'D',256},{6,256},{'M',2},{'D',256},{2,256}}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end
  
   function createElem(iChannels,oChannels, nNodes, nStride) 
      local upElem = nn.Sequential()
      local h=16
      for i=1,nNodes do   
         local bottleneck = nn.Sequential()
         local shortcut = nn.Sequential()
         local stride = 1
         if i==1 then 
            stride = nStride 
         end
         bottleneck:add(nn.SelectTable(1))        
         bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels/h,1,1,1,1,0,0))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels/h))
         bottleneck:add(cudnn.ReLU(true))

         bottleneck:add(cudnn.SpatialConvolution(iChannels/h,iChannels/h,3,3,stride,stride,1,1))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels/h))
         bottleneck:add(cudnn.ReLU(true))
         bottleneck:add(cudnn.SpatialConvolution(iChannels/h,iChannels,1,1,1,1,0,0))
         bottleneck:add(cudnn.SpatialBatchNormalization(iChannels))
         --bottleneck:add(cudnn.ReLU(true))
         --bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels,1,1,1,1,0,0))
         --bottleneck:add(cudnn.SpatialConvolution(iChannels,iChannels,3,3,1,1,1,1))
         --bottleneck:add(cudnn.SpatialBatchNormalization(iChannels))
         
         shortcut:add(nn.SelectTable(1))
         if stride>1 then
            shortcut:add(cudnn.SpatialMaxPooling(stride,stride))
         end
         
         local sumElem;
         local hob;   
         if opt.shortCut == 'none' then
              --- Without short Cut
            sumElem = nn.Sequential():add(bottleneck):add(cudnn.ReLU(true))
         elseif opt.shortCut == 'max' then   
            ---With Short Cut
             hob = nn.ConcatTable():add(bottleneck):add(shortcut);
             sumElem = nn.Sequential():add(hob):add(nn.CMaxTable(false)):add(cudnn.ReLU(true))
         elseif opt.shortCut == 'softmax' then
             hob = nn.ConcatTable():add(bottleneck):add(shortcut);
             sumElem = nn.Sequential():add(hob):add(nn.CMaxTable(true)):add(cudnn.ReLU(true))
         elseif opt.shortCut == 'sum' then
             hob = nn.ConcatTable():add(bottleneck):add(shortcut);
             sumElem = nn.Sequential():add(hob):add(nn.CAddTable()):add(cudnn.ReLU(true))
         elseif opt.shortCut == 'pca' then
             hob = nn.ConcatTable():add(bottleneck):add(shortcut);
             sumElem = nn.Sequential():add(hob):add(nn.JoinTable(2))
             sumElem:add(cudnn.SpatialConvolution(2*iChannels,iChannels,1,1,1,1,0,0))
             sumElem:add(cudnn.SpatialBatchNormalization(iChannels)):add(cudnn.ReLU(true))
         end
         upElem:add(nn.ConcatTable():add(sumElem):add(nn.Identity()))
      end
      
      if opt.scaleAggregation == 'none' then
         ---- Res Net : Picking the last scale 
         upElem:add(nn.FlattenTable()):add(nn.SelectTable(1))
         upElem:add(cudnn.SpatialConvolution(iChannels,oChannels,1,1,1,1,0,0)) 
      elseif opt.scaleAggregation == 'pca' then
      ---- Reducing All the Scales
         upElem:add(nn.FlattenTable()):add(nn.JoinTable(2))
         upElem:add(cudnn.SpatialConvolution(iChannels*(nNodes+1),oChannels,1,1,1,1,0,0)) 
      elseif opt.scaleAggregation == 'sum' then
      ---- Adding all the scales
         upElem:add(nn.FlattenTable()):add(nn.CAddTable())
         upElem:add(cudnn.SpatialConvolution(iChannels,oChannels,1,1,1,1,0,0)) 
      elseif opt.scaleAggregation == 'max' then
      ---- Maxing all the scales
         upElem:add(nn.FlattenTable()):add(nn.CMaxTable(false))
         upElem:add(cudnn.SpatialConvolution(iChannels,oChannels,1,1,1,1,0,0)) 
      elseif opt.scaleAggregation == 'softmax' then
         upElem:add(nn.FlattenTable()):add(nn.CMaxTable(true))
         upElem:add(cudnn.SpatialConvolution(iChannels,oChannels,1,1,1,1,0,0)) 
      end


      upElem:add(cudnn.SpatialBatchNormalization(oChannels)):add(cudnn.ReLU(true))           
      upElem:add(nn.ConcatTable():add(nn.Identity()))
      
      return upElem
   end

   local features=nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,cfg[1][2],7,7,2,2,3,3));
   features:add(cudnn.SpatialBatchNormalization(cfg[1][2]));
   features:add(cudnn.ReLU(true))
   
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
              local nStride = v[4]
              local resElem = createElem(iChannels,oChannels,nNodes,nStride)
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
