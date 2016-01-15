
local SCM,parent = torch.class('nn.SpatialCMul', 'nn.Module')

function SCM:__init(nFeature)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   

      self.weight = torch.rand(nFeature)
      --self.bias = torch.rand(nFeature)
      self.gradWeight = torch.Tensor(nFeature):zero()
      --self.gradBias = torch.randn(nFeature)
      --self:reset()
   
end

function SCM:reset()
   self.weight:uniform()
   --self.bias:zero()
end

function SCM:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer2 = self.buffer2 or input.new()
   self.output = input:clone()
   
   
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.output:cmul(self.buffer)
      --self.buffer:repeatTensor(self.bias:view(1, nFeature, 1, 1),
      --                        nBatch, 1, iH, iW)
      --self.output:add(self.buffer)

   return self.output
end

function SCM:updateGradInput(input, gradOutput)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   self.gradInput:resizeAs(input):zero()
   
      self.gradInput:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.gradInput:cmul(gradOutput)

   return self.gradInput
end

function SCM:accGradParameters(input, gradOutput, scale)
      scale = scale or 1.0
      local nBatch = input:size(1)
      local nFeature = input:size(2)
      local iH = input:size(3)
      local iW = input:size(4)
      self.buffer2:resizeAs(input):copy(input)
      self.buffer2=self.buffer2:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.buffer2:sum(self.buffer, 3) -- sum over pixels
      self.gradWeight:add(scale, self.buffer2)
      

     -- self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
     -- self.buffer2:sum(self.buffer, 3)
     -- self.gradBias:add(self.buffer2) -- sum over mini-batch

end