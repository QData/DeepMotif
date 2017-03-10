require 'nn'


-- Returns a list of tensors for each line in the file
local function loadData(name, maxLoad)
    print(name)
    if maxLoad == 0 then maxLoad = 1000000000 end
    local data = seq.lines(name):take(maxLoad):copy()
    function data:size() return #data end
    return data
end

function table_invert(t)
   local s={}
   for k,v in pairs(t) do s[v]=k end
   return s
end

function createDatasetOneHot(typ, opt)
    if (not opt.size) then opt.size = 0 end
    if (not opt.batch_size) then opt.batch_size = 1 end

    local seqs = loadData( path.join(opt.data_dir, typ..'.fa'), opt.size)
    local size
    if opt.size == 0 then size = seqs:size() end
    local inputs = {}
    local outputs = {}
    local alphabet = opt.alphabet
    local rev_lookup = {}
    for i = 1,#alphabet do
      rev_lookup[alphabet:sub(i,i)] = i
    end

    local class_labels = table_invert(opt.class_labels_table)

    setmetatable(inputs, {__index = function(self, ind)
      local upper_limit = math.min((ind+opt.batch_size),(size/2))
      local batch_len = upper_limit-ind
      local seq_len = opt.seq_length
      -- len = #seqs[ind*2]
      matrix = torch.zeros(batch_len, seq_len):fill(#alphabet+1)
      local batch_dim = 1
      for b = ind,upper_limit-1 do
        local str=seqs[b*2] -- have to multiply by 2 because of the way the data is set up with >1 and then sequence
        for i = 1,math.min(#str,seq_len),1 do
          if rev_lookup[str:sub(i,i)] then
            matrix[batch_dim][i] = rev_lookup[str:sub(i,i)]
          end
        end
        batch_dim = batch_dim+1
      end
      return matrix
    end})


    setmetatable(outputs, {__index = function(self, ind)
      local upper_limit = math.min((ind+opt.batch_size),(size/2))
      local batch_len = upper_limit-ind
      local labels = torch.ones(batch_len)
      local batch_dim = 1
      for i = ind,upper_limit-1 do
        local line = seqs[(i*2)-1]:gsub(' ','')--get label from i*2 -1 (because of FASTA format)
        local label = line:split('>')[1]
        label = class_labels[label]
        labels[batch_dim] = torch.Tensor({label})
        batch_dim = batch_dim+1
      end
      return labels
    end})

    function inputs:size() return size/2 end
    function outputs:size() return size/2 end
    return {inputs=inputs, labels=outputs}
end
