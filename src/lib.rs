#![allow(dead_code)]

// the full supported PARAMS are J,K,L,M


// define each layer params
pub struct LayerParams {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

// Defines dense layer struct
// A: first layer dimension
// B: last layer dimension
pub struct Layer<const A:usize,const B:usize> {
    linear: bool,
    classifier: bool,
    first_layer: Vec<[f32;A]>,
    last_layer: Vec<[f32;B]>,
    hiddens: Vec<LayerParams>,
}

// A: self-attention kv cache size
pub struct SelfAttentionParams<const A:usize> {
    key: [[f32;A];A],
    query: [[f32;A];A],
    value: [[f32;A];A],
}

// A: self-attention W
// B: self-attention H
pub struct SelfAttention<const A: usize,const B: usize> {
    weights: [[f32;A];B],
    value_vecs: [[f32;A];B],
    vec_key_matrix:[[[f32;A];A];B],
    vec_query_matrix: [[[f32;A];A];B],
    params: SelfAttentionParams<B>,
}

// A: self-attention W
// B: self-attention H
// C: head num
pub struct MultiHeadedAttentionParams<const A: usize,const B: usize,const C: usize> {
    heads: [SelfAttention<A,B>;C],
    linear: Layer<A,B>,
}

pub struct AddAndNorm<const A: usize,const B: usize> {
    original_input: [[f32;A];B],
    modified_input: [[f32;A];B],
}

// A: self-attention W
// B: self-attention H
// C: head num
pub struct EncoderBlockParams<const A: usize,const B: usize,const C: usize,> {
	multi_headed: MultiHeadedAttentionParams<A,B,C>,
    feed_forward: Layer<A,B>,
}

pub struct EncoderBlock<const A: usize,const B: usize,const C: usize> {
	add_and_norm: AddAndNorm<A,B>,
    params: EncoderBlockParams<A,B,C>,
}

// A,B,C: encoder size related
// D: encoder num
pub struct TransformerParams<const A: usize,const B: usize,const C: usize,const D: usize,> {
	encoder_blocks: [EncoderBlock<A,B,C>;D],
}

pub struct MyTransformer {
	input:Vec<u8>,
	output:f32,
	params:TransformerParams<500,1,5,4>,
}

impl MyTransformer {

	pub fn eval() {
		let bytes_usize:usize = std::mem::size_of::<MyTransformer>();
		println!("the size of my transformer(without hidden layers) is {:}(bytes) | {:}(Kbytes) | {:}(MBytes)",bytes_usize,bytes_usize >> 10,bytes_usize >> 20);
	}
}


