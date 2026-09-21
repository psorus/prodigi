import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, num_token_id=4, d_model=256, nhead=4, num_layers=3, max_seq_len=32, attribute_count=4, num_head=True, token_head=True, int_head=True, bool_head=True, token_count=200, int_count=120, num_cls=1, deep_num=True, use_encoder=True):
        super().__init__()
        self.num_token_id = num_token_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_head = num_head
        self.token_head = token_head
        self.int_head = int_head
        self.bool_head = bool_head
        self.attribute_count = attribute_count
        self.token_count = token_count
        self.int_count = int_count
        self.num_cls = num_cls
        self.use_encoder = use_encoder

        #print("initialized with num_cls", num_cls)
        #exit()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if deep_num:
            self.float_embedding = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model)
            )
        else:
            self.float_embedding = nn.Sequential(
                nn.Linear(1, d_model)
            )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        if self.use_encoder: 
            # Separate Encoder and Decoder Modules
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.attribute_decoder=nn.Linear(d_model*num_cls, attribute_count)
        
        # Reconstruction Projection Output Head
        if self.num_head:
            self.to_float = nn.Linear(d_model, 1)
        if self.token_head:
            self.to_token = nn.Linear(d_model, token_count)
        if self.int_head:
            self.to_int = nn.Linear(d_model, int_count)
        if self.bool_head:
            self.to_bool = nn.Linear(d_model, 2)

    def forward(self, input_ids, numerical_values, attention_mask, template, template_mask):
        # 1. Embed & Mix Inputs for the Encoder
        text_vectors = self.token_embedding(input_ids)
        float_vectors = self.float_embedding(numerical_values)
        is_float_mask = (input_ids == self.num_token_id).unsqueeze(-1)
        
        # Interleave tokens and floats
        x = torch.where(is_float_mask, float_vectors, text_vectors)
        x = self.pos_encoder(x)
        
        # 2. Encode sequence into the latent bottleneck
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded_sequence = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # Extract ONLY the CLS vector at index 0
        cls_latent = encoded_sequence[:, 0:self.num_cls, :] 

        #print("cls_latent shape", cls_latent.shape)
        #exit()

        cls_flat=cls_latent.view(cls_latent.size(0), -1)  # Flatten if num_cls > 1
        #print("cls_flat shape", cls_flat.shape)

        #exit()

        attributes=self.attribute_decoder(cls_flat).squeeze(1)
        #print("attributes shape", attributes.shape)
        
        
        # 3. Setup Decoder Template Input (No continuous injection to prevent cheating)
        template_vectors = self.token_embedding(template)
        decoder_input = self.pos_encoder(template_vectors) 
        
        # Handle template padding mask
        tgt_key_padding_mask = (template_mask == 0) if template_mask is not None else None
        
        # 4. Decode using template tokens via cross-attention with the bottleneck representation
        decoded_space = self.decoder(
            tgt=decoder_input,
            memory=cls_latent, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        #print("decoded_space shape", decoded_space.shape)
        
        # 5. Project directly to scalar continuous predictions
        predicted_floats=None
        if self.num_head:
            predicted_floats = self.to_float(decoded_space)
        predicted_tokens=None
        if self.token_head:
            predicted_tokens = self.to_token(decoded_space)
        predicted_ints=None
        if self.int_head:
            predicted_ints = self.to_int(decoded_space)
        predicted_bools=None
        if self.bool_head:
            predicted_bools = self.to_bool(decoded_space)
        
        ret={}
        if self.num_head:
            ret['float']=predicted_floats
        if self.token_head:
            ret['token']=predicted_tokens
        if self.int_head:
            ret['int']=predicted_ints
        if self.bool_head:
            ret['bool']=predicted_bools
        ret["attributes"]=attributes
        return ret

    @torch.no_grad()
    def encode(self, input_ids, numerical_values, attention_mask=None):
        """
        Compresses a batch of multi-modal sequences into their corresponding
        CLS latent bottleneck vectors.

        Args:
            input_ids: Tensor shape [Batch, Seq_Len]
            numerical_values: Tensor shape [Batch, Seq_Len, 1]
            attention_mask: Tensor shape [Batch, Seq_Len] (1 for real, 0 for padding)

        Returns:
            cls_latent: Tensor shape [Batch, 1, d_model]
        """
        self.eval()

        text_vectors = self.token_embedding(input_ids)
        float_vectors = self.float_embedding(numerical_values)
        is_float_mask = (input_ids == self.num_token_id).unsqueeze(-1)

        # Merge tokens and numbers
        x = torch.where(is_float_mask, float_vectors, text_vectors)
        x = self.pos_encoder(x)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded_sequence = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Extract the bottleneck
        cls_latent = encoded_sequence[:, 0:self.num_cls, :]
        cls_flat=cls_latent.view(cls_latent.size(0), -1)  # Flatten if num_cls > 1
        return cls_flat

    #@torch.no_grad()
    def decode(self, cls_flat, template_ids, template_mask=None, evalmode=True):
        """
        Fills out a provided template with continuous variables in a single parallel step.
        replaces your old auto-regressive loop structures completely.

        Args:
            cls_latent: [B, 1, d_model] representation matrix from the encoder.
            template_ids: [B, T_Len] structural token template matrix.
            template_mask: [B, T_Len] attention mask for the template sequence layout.

        Returns:
            predicted_floats: [B, T_Len, 1] fully populated continuous tensor array.
        """
        if evalmode:self.eval()

        attributes=self.attribute_decoder(cls_flat).squeeze(1)

        cls_latent=cls_flat.view(cls_flat.size(0), self.num_cls, self.d_model)  # Reshape back to [B, num_cls, d_model]

        # Build pure structure vectors from the template (withholding true float leaks)
        template_vectors = self.token_embedding(template_ids)
        decoder_input = self.pos_encoder(template_vectors)

        # Handle padding masks
        tgt_key_padding_mask = (template_mask == 0) if template_mask is not None else None


        # Bidirectional generation using the full structural view
        decoded_space = self.decoder(
            tgt=decoder_input,
            memory=cls_latent,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # 5. Project directly to scalar continuous predictions
        predicted_floats=None
        if self.num_head:
            predicted_floats = self.to_float(decoded_space)
        predicted_tokens=None
        if self.token_head:
            predicted_tokens = self.to_token(decoded_space)
        predicted_ints=None
        if self.int_head:
            predicted_ints = self.to_int(decoded_space)
        predicted_bools=None
        if self.bool_head:
            predicted_bools = self.to_bool(decoded_space)
        
        ret={}
        if self.num_head:
            ret['float']=predicted_floats
        if self.token_head:
            ret['token']=predicted_tokens
        if self.int_head:
            ret['int']=predicted_ints
        if self.bool_head:
            ret['bool']=predicted_bools
        ret["attributes"]=attributes
        return ret




if __name__ == "__main__":
    MOCK_VOCAB_SIZE = 150
    NUM_TOKEN_ID = 4  
    BATCH_SIZE = 8
    SEQ_LEN = 32

    # Instantiate model
    model = TransformerAutoencoder(
        vocab_size=MOCK_VOCAB_SIZE,
        num_token_id=NUM_TOKEN_ID,
        max_seq_len=SEQ_LEN
    )

    # Simulate passing identical data layout to both inputs
    dummy_input_ids = torch.randint(0, MOCK_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dummy_numerical_values = torch.randn(BATCH_SIZE, SEQ_LEN, 1)
    dummy_attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN)

    # Calling with duplicate structures as specified
    floats = model(
        input_ids=dummy_input_ids, 
        numerical_values=dummy_numerical_values, 
        attention_mask=dummy_attention_mask,
        template=dummy_input_ids,
        template_mask=dummy_attention_mask
    )

    print("--- MODEL LAYOUT VERIFIED ---")
    print("Predicted Floats Shape (Expected [8, 32, 1]):", floats.shape)
