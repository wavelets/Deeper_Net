local BinarySpatialConvolution, parent = torch.class('nn.BinarySpatialConvolution', 'nn.Module')

function BinarySpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   self.weightBuffer = self.weight.new():resizeAs(self.weight)
   --self.biasBuffer = self.bias.new():resizeAs(self.bias)


   self:reset()
end

function BinarySpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function binarizeWeight(self)
   self.weightBuffer:copy(self.weight);
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   self.weightMeanabs = self.weight:norm(1,2):div(self.nInputPlane * self.kH * self.kW);
   self.weight:sign()--:cmul(self.weightMeanabs:expand(self.weight:size()));
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end
local function binarizeInput(input,self)
   local convparams = {} 
   convparams.nInputPlane = 1
   convparams.nOutputPlane = 1
   convparams.kW = self.kW
   convparams.kH = self.kH
   convparams.dW = self.dW
   convparams.dH = self.dH
   convparams.padW = self.padW
   convparams.padH = self.padH
   convparams.weight = torch.CudaTensor(1, self.kH*self.kW):fill(1/(self.kH*self.kW))
   convparams.bias = torch.CudaTensor(1):zero()
   convparams.finput = convparams.weight.new()
   convparams.fgradInput =  convparams.weight.new()
   convparams.output =  convparams.weight.new()
 
   if not self.inputBuffer then
      self.inputBuffer = input.new():resizeAs(input);
   end
   self.inputBuffer:copy(input);

   local inputMeanabs = input:norm(1,2):div(self.nInputPlane)    
   self.outputMeanabs = inputMeanabs.nn.SpatialConvolutionMM_updateOutput(convparams,inputMeanabs)
   
   input:sign()
end
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function BinarySpatialConvolution:updateOutput(input)
   backCompatibility(self)
   binarizeWeight(self)
   binarizeInput(input,self)
   --viewWeight(self)
   input = makeContiguous(self, input)

   local out = input.nn.SpatialConvolutionMM_updateOutput(self, input)
   local si = out:size()
   local coefs = torch.cmul(self.outputMeanabs:repeatTensor(1,self.nOutputPlane,1,1),self.weightMeanabs:view(1,self.nOutputPlane,1,1):repeatTensor(si[1],1,si[3],si[4]));
   --coefs:sqrt()
   --local coefs = self.weightMeanabs:view(1,self.nOutputPlane,1,1):repeatTensor(si[1],1,si[3],si[4]);
   --local coefs = self.outputMeanabs:repeatTensor(1,self.nOutputPlane,1,1)


   out:cmul(coefs);
   --self.weight:copy(self.weightBuffer)
   self.weight:cmul(self.weightMeanabs:expand(self.weight:size()));
   unviewWeight(self)
   
   --s = input:size()
   --input:copy(self.inputBuffer)--:cmul(self.inputMeanabs:view(s[1],1,s[3],s[4]):repeatTensor(1,s[2],1,1))
   return out
end

function BinarySpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      backCompatibility(self)
      viewWeight(self)
      input, gradOutput = makeContiguous(self, input, gradOutput)
      local out = input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
      unviewWeight(self)
      return out
   end
end

function BinarySpatialConvolution:accGradParameters(input, gradOutput, scale)
   backCompatibility(self)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   viewWeight(self)
   local out = input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
   unviewWeight(self)
   self.weight:copy(self.weightBuffer)
   return out
end

function BinarySpatialConvolution:type(type,tensorCache)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function BinarySpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end