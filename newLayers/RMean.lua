local RMean, parent = torch.class('nn.RMean', 'nn.Module')

function RMean:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self._gradInput = torch.Tensor()
end

function RMean:_getPositiveDimension(input)
   local dimension = self.dimension
   return dimension
end

function RMean:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   self.output:mean(input, dimension)
   return self.output
end

function RMean:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:mul(1/input:size(dimension))
   self.gradInput = self.gradInput:expandAs(input)
   return self.gradInput
end