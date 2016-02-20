--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                by Sergey Ioffe, Christian Szegedy
   This implementation is useful for inputs coming from convolution layers.
   For Non-convolutional layers, see BatchNormalization.lua
   The operation implemented is:
   y =     ( x - mean(x) )
   -------------------- * gamma + beta
   standard-deviation(x)
   where gamma and beta are learnable parameters.
   The learning of gamma and beta is optional.
   Usage:
   with    learnable parameters: nn.BatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(0 [,eps] [,momentum])
   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5
   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentup of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--
local BN,parent = torch.class('nn.DepthBatchNormalization', 'nn.Module')

function BN:__init(nFeature, eps, momentum)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ' ..
          'Give 0 for no affine transform')
   self.eps = eps or 1e-5
   self.train = true
   self.centering = true;
   self.momentum = momentum or 0.1

   self.running_mean = torch.Tensor()
   self.running_std = torch.Tensor()
   if nFeature > 0 then self.affine = true end
   self.affine = false;
   if self.affine then
      self.weight = torch.Tensor(nFeature)
      self.bias = torch.Tensor(nFeature)
      self.gradWeight = torch.Tensor(nFeature)
      self.gradBias = torch.Tensor(nFeature)
      self:reset()
   end
end

function BN:reset()
   self.weight:uniform()
   self.bias:zero()
end

function BN:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer2 = self.buffer2 or input.new()
   self.centered = self.centered or input.new()
   self.norm = self.norm or input.new()
   self.centered:resizeAs(input)
   self.std = self.std or input.new()
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)
   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)
   if self.centering == false then 
    self.centered:copy(input)
  else

     if self.train == false then
        assert(self.running_mean:nDimension() ~= 0,
               'Module never run on training data. First run on some training data before evaluating.')
        self.buffer:repeatTensor(self.running_mean:view(1, nFeature, 1, 1), nBatch, 1, iH, iW)
        self.centered:add(input,-1, self.buffer)
     else -- training mode
        if self.running_mean:nDimension() == 0 then
           self.running_mean:resize(nFeature):zero()
        end
        -- calculate mean over mini-batch, over feature-maps
        local in_folded = input:view(nBatch, nFeature, iH * iW)
        self.buffer:mean(in_folded, 1)
        self.buffer2:mean(self.buffer, 3)
        self.running_mean:mul(1 - self.momentum):add(self.momentum, self.buffer2) -- add to running mean
        self.buffer:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                                 nBatch, 1, iH, iW)

        -- subtract mean
        self.centered:add(input, -1, self.buffer)                  -- x - E(x)
     end 
   end 
      -- self.buffer:resizeAs(input) 
      -- self.buffer:copy(self.centered)--:cmul(self.buffer)          -- [x - E(x)]^2
      -- local buf_folded = self.buffer:view(nBatch,nFeature,iH*iW)
      -- self.norm= buf_folded:norm(1,2):mul(1/(nFeature)):pow(-1/2) 
      -- self.norm:repeatTensor(self.norm:view(nBatch, 1, iH, iW),
      --                          1, nFeature, 1, 1)
     


            -- calculate standard deviation over mini-batch
      self.buffer:resizeAs(input)
      self.buffer:copy(self.centered):cmul(self.buffer)          -- [x - E(x)]^2
      local buf_folded = self.buffer:view(nBatch,nFeature,iH*iW)
      self.std:mean(buf_folded, 2)
      self.std:add(self.eps):sqrt():pow(-1)      -- 1 / E([x - E(x)]^2)
      self.norm:repeatTensor(self.std:view(nBatch, 1, iH, iW),
                                1, nFeature, 1, 1)


      self.output:cmul(self.centered, self.norm)

   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.output:add(self.buffer)
   end

   return self.output
end

function BN:updateGradInput(input, gradOutput)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')

   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)
   
   --self.buffer2:cmul(self.centered,self.centered):cmul(self.norm:pow(-2))
   --self.gradInput:copy(self.norm:pow(1/2)):add(-1,self.buffer2):cmul(gradOutput) 
   --self.buffer2:cmul(self.centered,gradOutput):view(nBatch,nFeature*iH*iW):sum(2):view(nBatch,1,1,1):repeatTensor(1,nFeature,iH,iW):cmul(self.centered)
   --self.gradInput:cmul(self.norm:pow(2),gradOutput):add(-1,self.buffer2):cdiv(self.norm:pow(3/2))
   --self.gradInput:copy(gradOutput)

   self.gradInput:cmul(self.centered, gradOutput)
   local gi_folded = self.gradInput:view(nBatch, nFeature, iH * iW)
   
   --self.buffer2:mean(self.buffer:mean(gi_folded, 1), 3)
   --self.gradInput:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
   --                            nBatch, 1, iH, iW)
   self.buffer2:mean(gi_folded, 2)
   self.gradInput:repeatTensor(self.buffer2:view(nBatch, 1, iH, iW),
                                1, nFeature, 1, 1)
   

   self.gradInput:cmul(self.centered):mul(-1)
   self.gradInput:cmul(self.norm):cmul(self.norm)

   self.buffer:mean(gradOutput:view(nBatch, nFeature, iH*iW), 1)
   self.buffer2:mean(self.buffer, 3)
   self.buffer:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)

   --self.buffer2:mean(gradOutput:view(nBatch, nFeature, iH*iW), 2)
   --self.buffer:repeatTensor(self.buffer2:view(nBatch, 1, iH, iW),
   --                             1, nFeature, 1, 1)  
   

   self.gradInput:add(gradOutput):add(-1, self.buffer)
   self.gradInput:cmul(self.norm)
   
   --print(self.gradInput:mean())


   if self.affine then
      self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end

function BN:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      local nBatch = input:size(1)
      local nFeature = input:size(2)
      local iH = input:size(3)
      local iW = input:size(4)
      self.buffer2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer2 = self.buffer2:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.buffer2:sum(self.buffer, 3) -- sum over pixels
      self.gradWeight:add(scale, self.buffer2)

      self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
      self.buffer2:sum(self.buffer, 3)
      self.gradBias:add(scale, self.buffer2) -- sum over mini-batch
   end
end