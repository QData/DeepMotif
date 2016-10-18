require 'nn'


-- Returns a list of tensors for each line in the file
local function loadData(name, maxLoad)
  print(name)
    if maxLoad == 0 then maxLoad = 10000000000 end
    local data = seq.lines(name):take(maxLoad):copy()
    function data:size() return #data end
    return data
end


function createDatasetOneHot(typ, opt, data_size)
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

    setmetatable(inputs, {__index = function(self, ind)
        len = #seqs[ind*2]
        matrix = torch.zeros(opt.batch_size, len):fill(#alphabet+1)
        local k = 1
        for i = ind,math.min((ind+opt.batch_size-1),(size/2)) do
          -- have to multiple by 2 because of the way the data is set up with >1 and then sequence
          str=seqs[i*2]
            j = matrix:size(2)
            for i = #str,1,-1 do
              if j > 0 then
                if rev_lookup[str:sub(i,i)] then
                  matrix[k][j] = rev_lookup[str:sub(i,i)] --convert letter to onehot
                else
                  matrix[k][j] = #alphabet + 1 --unknown char
                end
                j = j - 1
              end
            end
          k = k+1
        end
        return matrix
    end})

    setmetatable(outputs, {__index = function(self, ind)
        label = torch.zeros(opt.batch_size)
        local k = 1
        for i = ind,math.min((ind+opt.batch_size-1),(size/2)) do
          --get label from i*2 -1 (because of FASTA format)
          str = torch.Tensor({seqs[(i*2)-1]:split('>')[1]})
          if str[1] == -1 or str[1] == 0 then str[1] = 2 end --need 1 or 2 label
          label[k] = str
          k = k+1
        end
        return label
    end})

    function inputs:size() return size/2 end
    function outputs:size() return size/2 end
    return {inputs=inputs, labels=outputs}
end
