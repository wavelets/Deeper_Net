local CMaxTable, parent = torch.class('nn.CMaxTable', 'nn.Module')

function CMaxTable:__init(soft)
   parent.__init(self)
   self.gradInput = {}
   self.gradMask  ={}
   self.soft = soft
end

function CMaxTable:updateOutput(input)
   
   local function normalize_exp(y)
      local x=y:clone()
      local m = x:mean();
      local min_x = x:min();
      local max_x = x:max();
      local max_abs = math.max(math.abs(min_x),math.abs(max_x));
      x:add(-m);
      x:div(max_abs+(1e-10)):exp()
      return x
   end
   
   self.output:resizeAs(input[1]):copy(input[1])
   self.expSum = input[1]:clone():zero()
   
   for i=2,#input do
      self.output:cmax(input[i])
   end
 
   if not self.soft then
      for i=1,#input do
         self.gradMask[i] = torch.eq(self.output,input[i])
      end
   else
      for i=1,#input do
         self.gradMask[i] = normalize_exp(input[i])
         self.expSum:add(self.gradMask[i]);         
      end
      for i=1,#input do
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