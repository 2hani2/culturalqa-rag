"""Authored held-out questions, checked against their downloaded source records."""
import csv,json
from pathlib import Path
from research_protocol import load_records,record_ref,canonical,digest
from structured_store import record_to_text
from project_paths import ROOT,RESEARCH

SPECS=[
('DOSA',2,'Which beach in Visakhapatnam is described as a venue for Navy air shows?',['Rushi konda beach']),
('DOSA',13,'What is the name of the fort near Tirupati in the artifact collection?',['chandragiri fort']),
('DOSA',19,'What are the dolls called whose heads move while their bodies remain still?',['Buttabomma']),
('DOSA',34,'Name the two-piece Assamese garment worn in Muga silk during Bihu.',['Mekhla Chador']),
('DOSA',67,'Which minimalist Bihu celebration takes place in October with prayers for a good harvest?',['kati bihu']),
('DOSA',74,'What names does the collection give for a rice wine from Assam?',['xaaj and rohi']),
('DIWALI',7,'What alternative style name is given to Pedana Kalamkari?',['Machilipatnam style']),
('DIWALI',8,'Which drawing tool is used for hand-painted Srikalahasti Kalamkari designs?',['kalam','pen','kalam (pen)']),
('DIWALI',12,'Which Indian state is associated with the Sherdukpen Shawl?',['Arunachal Pradesh']),
('DIWALI',20,'Which silk is also called Endi or Errandi and is produced without killing the silkworm?',['Eri Silk']),
('DIWALI',24,'What is the Assamese garment worn under the Mekhela and over the upper body?',['Riha']),
('DIWALI',37,'Which plant supplies the roots used to dye Aal-Dyed Handloom Fabrics?',['Morinda citrifolia']),
('DRISHTIKON',27,'Name the two rivers whose confluence at Devprayag forms the Ganges.',['Alaknanda-Bhagirathi','Alaknanda and Bhagirathi']),
('DRISHTIKON',1462,'What instruments are predominantly used during Bhuta Kola performances?',['Drum (Dolu) and Cymbals (Taal)']),
('DRISHTIKON',1676,'What is the Ladakhi dance performed with wooden shields and swords called?',['Shondol']),
('DRISHTIKON',2468,'Which state is associated with folk theatre about migration and caste and with the ancient educational centre Nalanda?',['Bihar']),
('DRISHTIKON',2817,'What is the southern Indian international township associated with forest conservation and a vision of harmony?',['Auroville']),
('DRISHTIKON',3195,'In which Indian state is the ancient university Nalanda located?',['Bihar']),
('Indian Cultural Dataset',3,'Who illustrated the 42-issue Amar Chitra Katha Mahabharata comic series?',['Dilip Kadam']),
('Indian Cultural Dataset',4,"Who was Sati's father in the story of Shiva and Sati?",['Daksha Prajāpati','Daksha']),
('Indian Cultural Dataset',26,'Who is named as the father of King Bharata in the Mahabharata account?',['Dushyanta']),
('Indian Cultural Dataset',48,'Who taught music and dance to Uttara before her marriage to Abhimanyu?',['Arjuna']),
('Indian Cultural Dataset',72,"What was Bhanumati's home kingdom before she married Duryodhana?",['Kalinga']),
('Indian Cultural Dataset',90,'Who is credited with writing Betaal Pachisi in the Vikram and Betaal collection?',['Somdev Bhatt'])]
IMAGE_SPECS=[
(27,'Find a photograph of the Devprayag river valley.','Alaknanda_devprayag.jpg'),
(1462,'Find a Bhuta Kola ritual performer wearing a large ornate gold costume.','bhuta_kola.jpg'),
(1676,'Find Shondol dancers performing outdoors in Ladakh.','Shondol.jpg'),
(1959,'Find a photograph of a traditional Indian wedding with a bride and groom.','thali_ceremony.jpg'),
(2817,'Find an aerial photograph of forest buildings with solar panels at Sadhana Forest.','Sadhna_Forest.jpg'),
(3195,'Find a photograph of ancient Nalanda red-brick ruins.','Nalanda_Mahavihara_01.jpg')]

def build():
    records=load_records();lookup={record_ref(r):r for r in records};split=json.loads((RESEARCH/'split_manifest.json').read_text())
    specs=list(SPECS)
    ids=[8,96,199,296,400,504]
    forms=['How is {} written in the Hindi translation field?','Give the Hindi wording for the dish {}.',
           'What Hindi translation is supplied for {}?','How does the collection translate the festival name {} into Hindi?',
           'What is the Hindi rendering of the landmark {}?','How is the sports-person name {} represented in Hindi?']
    for rid,form in zip(ids,forms):
        r=lookup[f'SCALE#{rid}'];specs.append(('SCALE',rid,form.format(r['Item']),[r['Translation']]))
    queries=[]
    for i,(ds,rid,q,answers) in enumerate(specs,1):
        ref=f'{ds}#{rid}';r=lookup[ref];text=record_to_text(r)
        assert split['assignments'][ref]['fold']=='test',ref
        assert canonical(q)!=canonical(r.get('question','')),ref
        assert canonical(answers[0]) in canonical(text),(ref,answers[0])
        pos=text.casefold().find(answers[0].casefold());evidence=text[max(0,pos-100):pos+len(answers[0])+160]
        queries.append({'id':f'test_text_{i:02}','task':'text','question':q,'target_dataset':ds,'relevant_records':[ref],
            'expected_answers':answers,'evaluation_role':'group_heldout_source_verified','gold_record_ref':ref,
            'label_source':f'raw_records_cache.json {ref}; answer and support checked against the downloaded record',
            'source_text_sha256':digest(text),'source_evidence':evidence,'source_verified':True,'human_reviewed':False,
            'relevance_complete':False,'fold':'test','modality':'image_associated' if r.get('image_url') else 'text_only',
            'question_author':'Authored for evaluation independently of the training anchors'})
    for i,(rid,q,image) in enumerate(IMAGE_SPECS,1):
        ref=f'DRISHTIKON#{rid}';assert split['assignments'][ref]['fold']=='test'
        relative='data/raw/DRISHTIKON/All_Images_formatted_final_dataset/'+image
        queries.append({'id':f'test_image_{i:02}','task':'image','question':q,'target_dataset':'DRISHTIKON',
            'relevant_images':[relative],'gold_record_ref':ref,'evaluation_role':'group_heldout_source_verified',
            'label_source':'Downloaded image directly inspected; one verified positive image','source_verified':True,
            'visual_content_verified':True,'human_reviewed':False,'relevance_complete':False,'fold':'test','modality':'image_associated',
            'annotation_note':'Native Pongal QA is attached to a wedding photo; this image query uses the actual wedding content.' if rid==1959 else ''})
    doc={'description':'Thirty authored text questions and six visually checked image questions from test groups. Source-verified positives; not exhaustive relevance judgements or independent human expert review.',
         'split_manifest':'research/split_manifest.json','queries':queries}
    for path in (ROOT/'validation_queries.json',RESEARCH/'evaluation_questions.json'):path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    with (RESEARCH/'annotation_review.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['query_id','question','accepted_answers','positive_evidence','source_evidence','human_reviewed','reviewer','additional_relevant_refs','notes','relevance_complete'])
        for q in queries:w.writerow([q['id'],q['question'],json.dumps(q.get('expected_answers',[]),ensure_ascii=False),'; '.join(q.get('relevant_records',q.get('relevant_images',[]))),q.get('source_evidence',q.get('annotation_note','')),False,'','','',False])
    print('[evaluation] 36 held-out questions saved with source evidence; human-review flags remain false.')
if __name__=='__main__':build()
