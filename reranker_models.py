"""MoE and parameter-matched single-network fusion, with provenance envelopes."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from moe_fusion import MoEFusion
from project_paths import CHECKPOINTS

class SimpleFusion(nn.Module):
    def __init__(self,text_dim=768,img_dim=512,proj_dim=256,hidden_dim=1152):
        super().__init__()
        self.text_proj=nn.Linear(text_dim,proj_dim);self.img_proj=nn.Linear(img_dim,proj_dim)
        self.network=nn.Sequential(nn.Linear(proj_dim*2,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,proj_dim))
    def forward(self,text,image):return self.network(torch.cat([self.text_proj(text),self.img_proj(image)],-1))
    @torch.no_grad()
    def score_relevance(self,qt,qi,dt,di):return F.cosine_similarity(self(qt,qi),self(dt,di),dim=-1)

def sha256_file(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def source_hash():
    return {name:sha256_file(Path(__file__).parent/name) for name in ('moe_fusion.py','reranker_models.py','research_train.py','research_protocol.py','reader.py','retrieval.py')}
def create_model(kind,seed=42):
    torch.manual_seed(seed)
    if kind=='moe':return MoEFusion()
    if kind=='simple':return SimpleFusion()
    raise ValueError(f'Unknown model kind: {kind}')
def save_model(model,kind,path,provenance):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format_version':1,'model_kind':kind,'state_dict':model.state_dict(),'provenance':provenance},path)
    sidecar={**provenance,'checkpoint_sha256':sha256_file(path),'model_kind':kind}
    path.with_suffix('.provenance.json').write_text(json.dumps(sidecar,indent=2))
def load_model(kind,path=None):
    path=Path(path or CHECKPOINTS/f'{kind}_fusion.pt')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('model_kind')!=kind or 'provenance' not in payload:raise ValueError('Use a newly trained research checkpoint with documented provenance.')
    model=create_model(kind);model.load_state_dict(payload['state_dict']);model.eval()
    return model,{**payload['provenance'],'checkpoint_sha256':sha256_file(path),'path':str(path)}
