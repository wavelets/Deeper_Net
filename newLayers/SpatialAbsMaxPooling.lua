local SpatialAbsMaxPooling, parent = torch.class('nn.SpatialAbsMaxPooling', 'nn.Module')

function SpatialAbsMaxPooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.padW = padW or 0
   self.padH = padH or 0

   self.ceil_mode = false
   self.indices = torch.Tensor()
end

function SpatialAbsMaxPooling:ceil()
  self.ceil_mode = true
  return self
end

function SpatialAbsMaxPooling:floor()
  self.ceil_mode = false
  return self
end

function SpatialAbsMaxPooling:updateOutput(input)
   -- backward compatibility
   self.ceil_mode = self.ceil_mode or false
   self.padW = self.padW or 0
   self.padH = self.padH or 0
   local absinput = torch.abs(input)
   input.nn.SpatialMaxPooling_updateOutput(self, absinput)
   si = input:size();
   so = self.output:size()
   local ii = torch.view(self.indices,so[1],so[2],so[3]*so[4]):long();
   local temp_in = input:clone():view(si[1],si[2],si[3]*si[4]);
   local temp_out = self.output.new():resizeAs(self.output):view(so[1],so[2],so[3]*so[4]);
   
   for i=1,so[1] do
      for j= 1,so[2] do
          temp_out[i][j] = temp_in[i][j]:index(1,ii[i][j]):clone()
       end
    end
    self.output:copy(temp_out:view(self.output:size()))

   return self.output
end

function SpatialAbsMaxPooling:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialAbsMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end

function SpatialAbsMaxPooling:__tostring__()
   local s =  string.format('%s(%d,%d,%d,%d', torch.type(self),
                            self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'

   return s
end