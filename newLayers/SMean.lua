local SMean, parent = torch.class('nn.SMean', 'nn.Module')

function SMean:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
  
end

function SMean:updateOutput(input)
   self.output = input:clone();
   self.output:add(-1,input:mean(self.dimension):expandAs(input))
   return self.output
end

function SMean:updateGradInput(input, gradOutput)
   
    self.gradInput = gradOutput;
   return self.gradInput
end