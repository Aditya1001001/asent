# %%

import spacy
import asent
from ramda import *
# create spacy pipeline
nlp = spacy.load('en_core_web_sm', disable=['ner'])

#%%
# add the rule-based sentiment model
nlp.add_pipe("asent_wordnet_v1")

# %%
def get_sdp_path(doc, subj, obj, lca_matrix):
  lca = lca_matrix[subj, obj]
  # print(lca_matrix)
  # print(doc[subj], doc[obj])

  current_node = doc[subj]
  subj_path = [current_node]
  if lca != -1:
    if lca != subj:
      while current_node.head.i != lca:
        current_node = current_node.head
        subj_path.append(current_node)
      subj_path.append(current_node.head)
  current_node = doc[obj]
  obj_path = [current_node]
  if lca != -1:
    if lca != obj:
      while current_node.head.i != lca:
        current_node = current_node.head
        obj_path.append(current_node)
      obj_path.append(current_node.head)

  return subj_path + obj_path[::-1][1:]

# %%
from spacy.tokens import Token
Token.set_extension("sentiments", default=[])
Token.set_extension("sentiment", default=[])


# text = "Microsoft was GREAT startup selling stuff to stupid IBM"
# %%
text = "Quality of Xiaomi products was not bad, but it slightly degraded over time, while Samsung is getting better"
# text = "Quality of Xiaomi products is not bad,  while their competitors are getting better"
# text = "Quality of Xiaomi products is not that bad, while its getting better"
doc = nlp(text)

# %%
print(doc)

# %%
# asent.visualize(doc, style='prediction')

#TODO: https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt

# %%
T = [doc[2], doc[16]]
print(T)


# %%
for w in doc:
    candidates = []
    for t in T:
        if w._.polarity.polarity:

            # print(t.i)
            p = get_sdp_path(doc, t.i, w.i, doc.get_lca_matrix())

            has_no_link_through_other_token = is_empty(intersection(p, difference(T, [t])))
            token_in_link = t in p
            # print(w, t, t.head in p, p)

            if token_in_link and has_no_link_through_other_token:
                total_length = 0
                step = p[0]
                for link in p[1:]:
                    total_length += abs(step.i - link.i)
                    step = link
                if total_length != 0:
                    candidates.append([t, total_length, p])

                # print(t, w, token_in_link, has_no_link_through_other_token, p, total_length)
    if len(candidates):
        best_candidate = path([0,0], sort_by(prop(1), candidates))
        best_candidate._.sentiments.append(w._.polarity)

# %%
for t in T:
    t._.sentiment = mean(pluck('polarity', t._.sentiments))
    print(t, t._.sentiment)

    # t._.sentiment.push(w._.polarity)
            # print(w, w._.polarity.polarity, p, map(lambda x: x.tag_, p), list(w.subtree))
            # print(w, w._.polarity.polarity, sort_by(lambda x: x.i, get_sdp_path(doc, t.i, w.i, doc.get_lca_matrix())))

# %%
# for w in doc:
#     print(w, w._.sentiment)

# %%
spacy.displacy.render(doc, style='dep')

# %%
asent.visualize(doc
, style="analysis"
)

# %%
