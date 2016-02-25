function createModel(nGPU)
   require 'cudnn'

   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997

   local function MeanSub(dim,nFeatures)
         local C= nn.ConcatTable()
         C:add(nn.Identity())
         C:add(nn.Sequential():add(nn.RMean(dim)):add(nn.Replicate(nFeatures,dim,4)))
         return nn.Sequential():add(C):add(nn.CSubTable())
   end
   local function activation()
      local C= nn.Sequential()
      --C:add(cudnn.ReLU(true))
      --C:add(nn.AddConstant(-0.0001,true))
      --C:add(nn.SMean(2))
      C:add(nn.HardTanh())
      C:add(nn.HardSign())
      return C
   end

   local function ContConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
         
         local C= nn.Sequential()
          --C:add(cudnn.SpatialBatchNormalization(nInputPlane))
          --C:add(activation())
          C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
          C:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
          C:add(cudnn.ReLU(true))
          return C
   end
   local function MaxPooling(kW, kH, dW, dH, padW, padH)
    return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   end
   local function AvgPooling(kW, kH, dW, dH, padW, padH)
    local C = nn.Sequential()
    C:add(nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH))
    C:add(nn.HardSign())
    return C 
   end

local features = nn.Sequential()
   features:add(ContConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
   features:add(MaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(ContConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27  
   features:add(MaxPooling(3,3,2,2))                     --  27 ->  13
   features:add(ContConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(ContConvolution(384,384,3,3,1,1,1,1)) 
   features:add(ContConvolution(384,256,3,3,1,1,1,1)) 
   features:add(MaxPooling(3,3,2,2))           
   features:add(nn.SpatialDropout(opt.dropout))
   features:add(ContConvolution(256,4096,6,6))
   features:add(nn.SpatialDropout(opt.dropout))           
   features:add(ContConvolution(4096,4096,1,1)) 
   features:add(cudnn.SpatialConvolution(4096, nClasses,1,1))
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
 
   local model = features
   

   return model
end
