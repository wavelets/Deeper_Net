local CMaxTable, parent = torch.class('nn.CMaxTable', 'nn.Module')

function CMaxTable:__init(soft)
   parent.__init(self)
   self.gradInput = {}
   self.gradMask  ={}
   self.soft = soft
end

function CMaxTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   self.expSum = input[1]:clone():exp()
   for i=2,#input do
      self.output:cmax(input[i])
      self.expSum:add(torch.exp(input[i]))
   end
   if not self.soft then
      for i=1,#input do
         self.gradMask[i] = torch.eq(self.output,input[i])
      end
   else
      for i=1,#input do
         self.gradMask[i] = torch.exp(input[i])
         self.gradMask[i]:cdiv(self.expSum)
      end
   end
   return self.output
end

function CMaxTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
      self.gradInput[i]:copy(gradOutput)
      self.gradInput[i]:cmul(self.gradMask[i])
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end