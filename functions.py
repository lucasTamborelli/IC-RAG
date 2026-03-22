from pinecone import Pinecone
from collections import defaultdict
from dotenv import load_dotenv
import tiktoken
from typing import *
import os
import streamlit as sl
import json

load_dotenv()
pinecone_key = os.getenv("API_KEY_PINECONE")

def semantic_search(index_name, text_chunks, query, top_k, printer):
	
	pc = Pinecone(api_key = pinecone_key)
	if not pc.has_index(index_name):
		pc.create_index_for_model(
			name=index_name,
			cloud = 'aws',
			region = 'us-east-1',
			embed = {
				'model':'llama-text-embed-v2',
				'field_map':{'text': 'chunk_text'}
			}
		)
	records = [
		{
			"_id": f"chunk_{i+1}",
			"chunk_text": chunk,
			"category": "pdf_content"
		}
		for i, chunk in enumerate(text_chunks)
	]
	dense_index = pc.Index(index_name)
	dense_index.upsert_records('teste1', records)
	results = dense_index.search(
		namespace="teste1",
		query={
			"top_k": top_k,
			"inputs": {
				'text': query
			}
		}
	)
	context_texts = []
	for hit in results['result']['hits']:
			if printer:
				print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")	 
			context_texts.append(hit['fields']['chunk_text'])

	# dense_context = "\n\n".join(context_texts)
	# return dense_context
	return results  # <-- Return the full results dict


def keyword_search(index_name, text_chunks, query, top_k, printer):
	
	pc = Pinecone(api_key = pinecone_key)
	if not pc.has_index(index_name):
		pc.create_index_for_model(
			name=index_name,
			cloud="aws",
			region="us-east-1",
			# metric = "dotproduct",
			embed={
				"model": "pinecone-sparse-english-v0",
				"field_map": {"text": "chunk_text"}
			}
		)
	records = [
		{
			"_id": f"chunk_{i+1}",
			"chunk_text": chunk,
			"category": "pdf_content"
		}
		for i, chunk in enumerate(text_chunks)
	]
	sparse_index = pc.Index(index_name)
	sparse_index.upsert_records(namespace="pdf", records=records)
	results = sparse_index.search(
		namespace="pdf",
		query={
			"top_k": top_k,
			"inputs": {"text": query}
		}
	)
	context_texts = []
	for hit in results["result"]["hits"]:
		if printer:
			print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")	
	
	# sparse_context = "\n\n".join(context_texts)
	# return sparse_context
	return results  



def hybrid_search(dense_result, sparse_result, sparse_weight, top_k, printer):
	final_score = defaultdict(lambda: {"text": "", "dense": 0, "sparse": 0}) 

	for hit in dense_result["result"]["hits"]:
		_id = hit["_id"]
		final_score[_id]["text"] = hit["fields"]["chunk_text"]
		final_score[_id]["dense"] = hit["_score"]	


	for hit in sparse_result["result"]["hits"]:
		_id = hit["_id"]
		final_score[_id]["text"] = hit["fields"]["chunk_text"]
		final_score[_id]["sparse"] = hit["_score"]


	final = []
	for _id, vals in final_score.items():
		combined_score = sparse_weight * vals["sparse"] + (1 - sparse_weight) * vals["dense"]
		final.append({"_id": _id, "text": vals["text"], "score": combined_score})

	final = sorted(final, key=lambda x: x["score"], reverse=True)
	top_k_final = final[:top_k]
	
	for hit in top_k_final:
		if printer: 
			print(f"id: {hit['_id']} | score: {hit['score']:.2f} | text: {hit['text'][:500]}...")
	return top_k_final


def save_feedback(arq, query, answer, type):
	score = sl.session_state.get(f'avaliacao_{type}')
	feedback = {"Pergunta": query, "Resposta": answer, "Avaliacao": score}
	if os.path.isfile(arq):
		with open(arq, 'r', encoding='utf-8') as f:
			data = json.load(f)
	else:
		data = []
	data.append(feedback)
	with open(arq, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	print(f"Answers saved {score}")

@sl.fragment
def render_tab(query, type, answer, tokens):
	sl.markdown(f"**Tipo de resposta:** {type}")
	sl.markdown(f"**Resposta:**")
	sl.write(answer)
	sl.write("---")
	
	avaliacao = sl.radio(
		f'Avalie a resposta {type}:', 
		("1 (Muito Ruim)", "2", "3", "4", "5 (Muito Bom)"),
		key=f'eval_{type}', 
		index=None
	)
	
	if avaliacao:
		sl.session_state[f'avaliacao_{type}'] = avaliacao
		sl.success("Avaliação registrada!")
		save_feedback('feedback.json', query, answer, type)

	sl.text(f"Tokens: {tokens}")

def n_tokens(input: str, output: str, model_name: str):
	enc = tiktoken.get_encoding(model_name)
	tokens_in = enc.encode(input)
	tokens_out = enc.encode(output)
	return [len(tokens_in), len(tokens_out)]
