local SpatialSoftMaxPooling, parent = torch.class('nn.SpatialSoftMaxPooling', 'nn.Module')

function SpatialSoftMaxPooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.divide = true
end

function SpatialSoftMaxPooling:updateOutput(input)
   
   input:exp()
   input.nn.SpatialAveragePooling_updateOutput(self, input)
   input:log()
   self.output:log()
   -- for backward compatibility with saved models
   -- which are not supposed to have "divide" field
   if not self.divide then
     self.output:mul(self.kW*self.kH)
   end
   return self.output
end

function SpatialSoftMaxPooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SpatialAveragePooling_updateGradInput(self, input, gradOutput)
      -- for backward compatibility
      if not self.divide then
	self.gradInput:mul(self.kW*self.kH)
      end
      return self.gradInput
   end
end

function SpatialSoftMaxPooling:__tostring__()
   return string.format('%s(%d,%d,%d,%d)', torch.type(self),
         self.kW, self.kH, self.dW, self.dH)
end