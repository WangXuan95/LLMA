import os
import sys
from   io import BytesIO
import time
import torch
import transformers



class BinaryArithmeticCoder () :
    def __init__ (self, stream:BytesIO, is_encode:bool) :
        self.stream = stream
        self.is_encode = is_encode
        self.MASK= 0xFFFFFFFFFFFFFFFF      # 64 bits
        self.sft = 0
        self.low = 0
        self.hgh = self.MASK
    
    def codecUint32 (self, value:int=0) :  # input/output uint32 without compress
        if self.is_encode :
            val = value
            for _ in range(4) :
                new_byte = (val >> 24) & 0xFF
                val <<= 8
                self.stream.write(bytes([new_byte,]))
        else :
            val = 0
            for _ in range(4) :
                new_byte = self.stream.read(1)[0]
                val <<= 8
                val  |= new_byte
        return val
    
    def _normalize (self) :
        if self.is_encode :
            new_byte = self.low >> 56
            self.stream.write(bytes([new_byte,]))
        else :
            new_byte = self.stream.read(1)[0]
            self.sft <<= 8
            self.sft  &= self.MASK
            self.sft  |= new_byte
        self.low <<= 8
        self.low  &= self.MASK
        self.hgh <<= 8
        self.hgh  &= self.MASK
        self.hgh  |= 0xFF
    
    def start_decode (self) :
        if not self.is_encode :
            for _ in range(8) :
                self._normalize()
        self.low = 0
        self.hgh = self.MASK
    
    def flush_encode (self) :
        if self.is_encode :
            for _ in range(8) :
                self._normalize()
    
    def codec_bin (self, weight0, weight1, bin:bool) -> bool :
        mid = ((self.hgh - self.low) * weight0) // (weight0 + weight1)
        mid+= self.low
        
        assert (self.low <= mid) and (mid < self.hgh)
        
        if not self.is_encode :
            bin = (self.sft > mid)
        
        if bin :
            self.low = mid + 1
        else :
            self.hgh = mid
        
        while (self.low >> 56) == (self.hgh >> 56) :
            self._normalize()
        
        return bin
    
    def __len__ (self) :
        '''get encoded length (in bytes)'''
        return self.stream.tell()



class BinaryDivisionArithmeticCoder (BinaryArithmeticCoder) :
    def _codec_value_recur (self, hist:torch.Tensor, si, ei, value:int) -> int :
        if   si + 1 == ei :
            return si
        elif si + 1 <  ei :
            di = (si + ei) // 2
            bin = (value >= di)
            weight0 = hist[si:di].sum().item()
            weight1 = hist[di:ei].sum().item()
            bin = self.codec_bin(weight0, weight1, bin)
            if bin :
                return self._codec_value_recur(hist, di, ei, value)
            else :
                return self._codec_value_recur(hist, si, di, value)
        else :
            raise Exception('index overflow in BinaryDivisionArithmeticCoder')
            return None
    
    def _codec_value (self, hist:torch.Tensor, value:int) :
        return self._codec_value_recur(hist, 0, len(hist), value)



def getHistogramFromProbs (probs:torch.Tensor) -> torch.LongTensor :
    return (probs * (2**30) + 1.0001).int()
    



def LLMAcodecTokens (model, stream:BytesIO, token_ids:torch.Tensor=None, verify=False) :
    is_encode = (not (token_ids is None)) and (not verify)
    
    codec = BinaryDivisionArithmeticCoder(stream, is_encode)
    
    if is_encode :                                                # encode
        n_token = token_ids.shape[-1]
        codec.codecUint32(n_token)                                # write token number
        codec.codecUint32(token_ids[0,0].item())                  # write the first token ID
    elif not verify :                                             # decode
        n_token = codec.codecUint32()                             # read token number
        token_ids = torch.zeros([1, n_token], dtype=torch.int64)
        token_ids[0,0] = codec.codecUint32()                      # read the first token ID
    else :                                                        # verify
        n_token = codec.codecUint32()                             # read token number
        assert n_token == token_ids.shape[-1]                     # verify the token number
        assert token_ids[0,0] == codec.codecUint32()              # verify the first token ID
    
    codec.start_decode()
    
    N_TOKENS_IN_KVCACHE = 2048
    N_TOKENS_RE_PREFILL = N_TOKENS_IN_KVCACHE // 2
    
    kvcache = transformers.cache_utils.DynamicCache()             # An empty KVcache
    
    torch.manual_seed(114514)
    
    start_time = time.time()
    
    with torch.no_grad() :
        for pos in range(1, n_token) :                            # From the 2nd token to the last token, compress each token
            if kvcache.get_seq_length() >= N_TOKENS_IN_KVCACHE :
                kvcache = transformers.cache_utils.DynamicCache() # clear KVcache
                spos = pos - N_TOKENS_RE_PREFILL
                prev_ids = token_ids[:, spos:pos]                 # shape = (1, N_TOKENS_RE_PREFILL)
                kvcache_info = ('Clear KVcache, re-prefill using token%d~token%d' % (spos, pos-1))
            else :
                prev_ids = token_ids[:, pos-1:pos]                # shape = (1, 1)
                kvcache_info = ('KVcache length = %d' % (kvcache.get_seq_length(),))
            
            model_out = model.forward(
                prev_ids,
                use_cache=True,
                past_key_values=kvcache
            )
            
            pred_logits = model_out.logits[0, -1, :]
            pred_probs  = torch.nn.functional.softmax(pred_logits, dim=0)
            
            hist = getHistogramFromProbs(pred_probs)
            
            curr_id = token_ids[0, pos].item() if is_encode else 0
            curr_id = codec._codec_value(hist, curr_id)
            
            if verify :
                assert token_ids[0, pos] == curr_id
            
            token_ids[0, pos] = curr_id
            
            print('[%d/%d] prob=%.6f    ratio=%.4f B/token    speed=%.3f token/sec    %s' % (
                pos,
                n_token,
                pred_probs[curr_id],
                len(codec) / pos,
                pos / (time.time() - start_time + 0.000001),
                kvcache_info
            ))
    
    codec.flush_encode()
    
    return token_ids



if __name__ == '__main__' :
    MODE, TEXT_FNAME, LLMA_FNAME = sys.argv[1:4]
    
    assert MODE in ['-c', '-d', '--verify'], 'MODE must be -c, -d, or --verify'
    assert TEXT_FNAME.endswith('.txt')    , 'TEXT_FNAME must be .txt'
    assert LLMA_FNAME.endswith('.llma')   , 'TEXT_FNAME must be .llma'
    
    local_model_path = './model_params/Qwen/Qwen2.5-0.5B'
    
    print('Info: Load model and tokenizer from', local_model_path, '...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
    model = transformers.Qwen2ForCausalLM.from_pretrained(local_model_path)
    model.eval()
    
    print()
    
    if   (MODE == '-c') :                                  # mode=compress
        with open(TEXT_FNAME, 'rt') as fp_in_text :
            input_text = fp_in_text.read()
        
        print('Compress %s (%d B) -> %s' % (TEXT_FNAME, len(input_text), LLMA_FNAME))
        assert len(input_text) >= 8, 'text is too short to be compressed'
        
        token_ids = tokenizer(input_text, return_tensors='pt').input_ids
        print('Compress %s (%d B, %d tokens) -> %s' % (TEXT_FNAME, len(input_text), token_ids.shape[-1], LLMA_FNAME))
        
        with open(LLMA_FNAME, 'wb') as fp_out_llma :
            LLMAcodecTokens(model, fp_out_llma, token_ids) # LLMA compress
        
        llma_size = os.path.getsize(LLMA_FNAME)
        print('Compress %s (%d B, %d tokens) -> %s (%d B)' % (TEXT_FNAME, len(input_text), token_ids.shape[-1], LLMA_FNAME, llma_size))
        
    elif (MODE == '-d') :                                  # mode=decompress
        llma_size = os.path.getsize(LLMA_FNAME)
        print('Decompress %s (%d B) -> %s' % (LLMA_FNAME, llma_size, TEXT_FNAME))
        
        with open(LLMA_FNAME, 'rb') as fp_in_llma :
            token_ids = LLMAcodecTokens(model, fp_in_llma) # LLMA decompress
        
        print('Decompress %s (%d B) -> %s (%d tokens)' % (LLMA_FNAME, llma_size, TEXT_FNAME, token_ids.shape[-1]))
        
        output_text = tokenizer.batch_decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        print('Decompress %s (%d B) -> %s (%d tokens, %d B)' % (LLMA_FNAME, llma_size, TEXT_FNAME, token_ids.shape[-1], len(output_text)))
        
        with open(TEXT_FNAME, 'wt') as fp_out_text :
            fp_out_text.write(output_text)
    
    else :                                                 # mode=verify
        with open(TEXT_FNAME, 'rt') as fp_in_text :
            input_text = fp_in_text.read()
        
        token_ids = tokenizer(input_text, return_tensors='pt').input_ids
        
        llma_size = os.path.getsize(LLMA_FNAME)
        
        print('Verify %s (%d B) <-> %s (%d B)' % (TEXT_FNAME, len(input_text), LLMA_FNAME, llma_size))
        
        with open(LLMA_FNAME, 'rb') as fp_in_llma :
            LLMAcodecTokens(model, fp_in_llma, token_ids, verify=True)


