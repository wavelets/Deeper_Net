local SelfIndex, parent = torch.class('nn.SelfIndex', 'nn.Module')

function SelfIndex:__init(dimension, index)
    parent.__init(self)
    self.dimension = dimension
    self.index =index
    self.gradInput = self.gradInput
end

function SelfIndex:updateOutput(input)
    self.output:index(input, self.dimension, self.index)
    return self.output
end

function SelfIndex:updateGradInput(input, gradOutput)

    local gradInput = self.gradInput -- no gradient for the index variable
    gradInput:resizeAs(input):zero()
    gradInput:indexAdd(self.dimension, self.index, gradOutput)
    return self.gradInput
end