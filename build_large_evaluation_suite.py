"""100 labelled text questions and 20 directly inspected image targets.

Retains the previous benchmark; questions are authored from held-out source
facts, not training anchors. These are source-verified positives, not expert
review or exhaustive full-corpus relevance judgements.
"""
import collections,copy,csv,json
from pathlib import Path
from project_paths import ROOT,RESEARCH
from research_protocol import load_records,record_ref,canonical,digest,image_digest
from structured_store import record_to_text

NEW_TEXT=[
('DOSA',45,'What material is called Baith in Assamese?',['Cane']),
('DOSA',79,'Which dance does the collection associate with Assam tea cultivators?',['Jhumur Dance']),
('DOSA',100,'Which Bihar freedom fighter founded the Janata Party in the collection?',['jayaprakash narayan']),
('DOSA',109,'What Bihar folk song is sung by women to celebrate the birth of a child?',['sohar']),
('DOSA',157,'What name does the collection give to the heritage well in Patan, Gujarat?',['raan ki vav']),
('DOSA',180,'Which Gujarati newborn ritual involves feeding the baby jaggery water?',['Galthuthi']),
('DOSA',206,'What Haryana wedding ritual invites the maternal uncle to the wedding?',['Bhaat Nautna']),
('DOSA',221,'Which national park in Ormanjhi is named after a freedom fighter?',['Birsa Munda National Park']),
('DOSA',234,'What mined material helped Dhanbad become an industrial area?',['Coal']),
('DOSA',275,'Which Karnataka dish prepared for Ugadi has both sweet and savoury variants?',['kadbu']),
('DOSA',303,'Which alcoholic drink in Kerala is made using coconut?',['toddy']),
('DOSA',367,'What percussion instrument is used in Pala theatre?',['Khanjoni']),
('DOSA',443,'Which Jaipur monument is described as having many windows?',['hawa mahal']),
('DOSA',514,'What material are the vegetable-coloured blankets from Warangal made of?',['Cotton']),
('DIWALI',143,'What is the tie-dye technique used for Sambalpuri Ikat called?',['Bandha']),
('DIWALI',156,'What long robe is traditionally worn by Sikh men during religious ceremonies?',['Chola']),
('DIWALI',523,'During which festival is Rathwa Ni Gher performed?',['Holi']),
('DIWALI',1896,'Which high-altitude lake in Ladakh is described as known for birdlife?',['Tso Moriri']),
('DIWALI',2628,'What other name is given to Eid ul-Azha?',['Bakrid']),
('DIWALI',2390,'Which Odisha festival celebrates womanhood and the onset of the monsoon?',['Raja']),
('DIWALI',3497,'What Sikkim food consists of fermented bamboo shoot?',['Mesu']),
('DIWALI',4399,'How many pits are on the wooden board used for Vamana Guntalu?',['14']),
('DIWALI',4939,'Which deity is ceremonially married to the Tulsi plant in Tulsi Vivah?',['Vishnu']),
('DIWALI',5153,'What alternative name is given to the Bilaspuri dialect?',['Kahluri']),
('DIWALI',5738,'What traditional Tamil Nadu puppetry uses large wooden puppets controlled by rods and strings?',['Bommalattam']),
('DIWALI',6022,'Which necklace of black glass beads symbolizes marriage among Mangalorean Catholics?',['Pirduk']),
('DIWALI',7670,'What city contains Sana Konung, the Royal Palace of Manipur?',['Imphal']),
('DIWALI',7969,'Which two ingredients make the Solkadhi drink described in the collection?',['Kokum and coconut milk']),
('DRISHTIKON',35,'Which Uttarakhand temple is associated with medieval architecture in the collection?',['Kedarnath']),
('DRISHTIKON',95,'What dish is described as the State Food of Uttarakhand?',['Kafuli']),
('DRISHTIKON',176,'Which Nellore-region textile combines cotton and silk with jamdani weaving?',['Uppada']),
('DRISHTIKON',177,'What traditional festive clothing do young girls in Rayalaseema wear?',['Langa Voni']),
('DRISHTIKON',210,'Name the Araku Valley tribal dance with chain formations and rhythmic foot movements.',['Dhimsa']),
('DRISHTIKON',424,'Which culture makes rice-paste floor patterns during Lakshmi Puja?',['Bengali']),
('DRISHTIKON',715,'What alternative name is used for Manipuri dance?',['Jagoi dance']),
('DRISHTIKON',725,'Whom does the Kang festival in Manipur honor?',['Lord Krishna','Honoring Lord Krishna']),
('DRISHTIKON',743,'What Vishnu temple is located in Vishnupur?',['Vishnu Temple of Vishnupur']),
('DRISHTIKON',857,'Which spice is known as Chinese pepper in Naga cuisine?',['SichuanPepper','Sichuan pepper']),
('DRISHTIKON',859,'What Naga sweet is made from rice flour, bananas, and jaggery?',['KoatPitha','Koat Pitha']),
('DRISHTIKON',861,'Which pungent allium is paired with chili and ginger in Naga cooking?',['Garlic']),
('DRISHTIKON',864,'What method preserves meat for long periods in Naga kitchens?',['Smoking']),
('DRISHTIKON',1690,"Who helped defend Ladakh in 1948 and later became India's ambassador to Mongolia?",['Kushok Bakula Rinpoche']),
('Indian Cultural Dataset',20,"Who translated the collection's volume of The Ramayana of Valmiki?",['Hari Prasad Shastri']),
('Indian Cultural Dataset',22,"Who coauthored Traditional games in India: Their origin and status in progressive era with Dr. Rajesh Dhauta?",['Satish Gulia']),
('Indian Cultural Dataset',23,'Who wrote Human Sacrifice in Colonial Central India: Myth, Agency and Representation?',['Crispin Bates']),
('Indian Cultural Dataset',51,"What is the name of Rama's bow in the source overview?",['Sharanga']),
('Indian Cultural Dataset',52,'What is the illusionary duplicate of Sita abducted by Ravana called?',['Maya Sita']),
('Indian Cultural Dataset',61,'Who gave Karna his Kavacha armour at birth?',['Surya']),
('Indian Cultural Dataset',62,'What classroom subject is discussed in the article on folk tales of North East India?',['Environmental Studies']),
('Indian Cultural Dataset',68,'Who wrote the story The Trinity in the collection?',['Nisha Susan']),
('Indian Cultural Dataset',94,"In which city was Dr. Swati Kela's Traditional Indian Games presentation at the April 2024 World Forum held?",['Vancouver']),
('Indian Cultural Dataset',3,'Who edited the Amar Chitra Katha Mahabharata comic series?',['Anant Pai']),
('Indian Cultural Dataset',4,"What leaf did Sati eat during her penance before giving up food altogether?",['bilva leaf']),
('Indian Cultural Dataset',26,"Who is named as King Bharata's mother in the Mahabharata account?",['Shakuntala']),
('Indian Cultural Dataset',48,"Who was Uttara's husband in the Mahabharata?",['Abhimanyu']),
('Indian Cultural Dataset',72,"Who is named as Bhanumati's father in the source?",['Chitrangada']),
]

SCALE_IDS=[12,25,101,116,201,227,313,317,406,411,516,518,601,609]
IMAGE_SPECS=[
(4,'kedarnath.jpg','Find a photograph of a stone Himalayan temple surrounded by snow-covered mountains.','Stone temple, colorful entrance, snow-covered roof and mountain slopes.'),
(215,'borra_caves.jpg','Find an interior photograph of Borra Caves showing a lit rocky passage with metal railings.','Rocky cave passage, a walkway and railings, artificial yellow lighting.'),
(474,'Cellular_Jail.jpg','Find the front entrance of Cellular Jail with two yellow round towers and green balcony railings.','CELLULAR JAIL entrance text, paired yellow towers and green railings.'),
(637,'Amritsari_Chole_Bhature_01.jpg','Find a close-up of chickpea curry garnished with onion, coriander, and a lemon slice.','Bowl of chickpea curry; lemon, onion, coriander and spoon. No bhature bread is visible.'),
(1372,'Taj_Mahal.jpg','Find a frontal photograph of the white Taj Mahal dome and its minarets under a cloudy blue sky.','White domed monument, four minarets, frontal perspective and cloudy blue sky.'),
(1373,'agra_fort_01.jpg','Find the red sandstone entrance to Agra Fort with visitors walking across the entrance bridge.','Red sandstone gateway, rounded defensive walls, arched entrance and visitors.'),
(1384,'Hampi.jpg','Find the carved stone chariot at Hampi with large stone wheels and elephant sculptures.','Ornate stone chariot, carved wheels, stone elephants and surrounding ruins.'),
(1390,'jog_falls.jpg','Find a photograph of broad Jog Falls cascading down several channels between green cliffs.','Several large waterfalls, white spray and green vegetation.'),
(1397,'Mysore_Painting.jpg','Find an ornate Mysore-style painting of a seated deity playing a string instrument with gold decorative details.','Painted seated deity with string instrument, rich gold detailing and ornate frame.'),
(1681,'Sand_Mandala.jpg','Find a brightly coloured circular sand mandala with geometric squares and a central multicoloured flower.','Circular mandala, colored geometric motifs and a central flower pattern.'),
(1730,'qutub.jpg','Find a photograph of seated male musicians singing into microphones and playing harmonium under coloured stage lights.','Musicians seated in a row, microphones, harmonium and colored lights; the image is not a tower photograph.'),
(1784,'Agrasen_ki_Baoli_01.jpg','Find a photograph looking down the deep steps of Agrasen ki Baoli with arches along both side walls.','Deep rectangular stepwell, descending stairs and arched wall recesses.'),
(1790,'The_India_Gate_01.jpg','Find India Gate photographed at night with orange, white, and green lighting and an empty central road.','Lit memorial arch at night, colored lighting and central road with lamps.'),
(2126,'Vembanad_Lake.jpg','Find a photograph of Kerala backwaters with houseboats on the water and coconut palms along the shore.','Several houseboats, broad waterway and palm-lined banks.'),
]

def build():
    records=load_records();lookup={record_ref(r):r for r in records}
    split=json.loads((RESEARCH/'split_manifest.json').read_text())
    queries=copy.deepcopy(json.loads((ROOT/'validation_queries.json').read_text())['queries'])
    specs=list(NEW_TEXT)
    for i,rid in enumerate(SCALE_IDS):
        r=lookup[f'SCALE#{rid}']
        if i%2==0:q=f"What Hindi translation does the collection give for {r['Item']}?";answers=[r['Translation']]
        else:q=f"Which item corresponds to the Hindi wording '{r['Translation']}' in the collection?";answers=[r['Item']]
        specs.append(('SCALE',rid,q,answers))
    for number,(ds,rid,question,answers) in enumerate(specs,1):
        ref=f'{ds}#{rid}';r=lookup[ref];text=record_to_text(r)
        if split['assignments'][ref]['fold']!='test':raise ValueError(f'Not held out: {ref}')
        if canonical(question)==canonical(r.get('question','')):raise ValueError('Native question reused verbatim.')
        if canonical(answers[0]) not in canonical(text):raise ValueError(f'Accepted answer not supported: {ref} {answers}')
        position=text.casefold().find(answers[0].casefold())
        if position<0:position=0
        evidence=text[max(0,position-140):position+len(answers[0])+200]
        queries.append({'id':f'large_text_{number:03}','task':'text','question':question,'target_dataset':ds,
            'relevant_records':[ref],'expected_answers':answers,'gold_record_ref':ref,'fold':'test',
            'evaluation_role':'group_heldout_source_verified','source_verified':True,'human_reviewed':False,
            'relevance_complete':False,'source_text_sha256':digest(text),'source_evidence':evidence,
            'label_source':f'Downloaded source record {ref}; positive answer and quoted evidence checked',
            'question_author':'Authored from held-out source facts independently of the training-anchor questions',
            'modality':'image_associated' if r.get('image_url') else 'text_only'})
    for number,(rid,name,question,visual) in enumerate(IMAGE_SPECS,1):
        ref=f'DRISHTIKON#{rid}';r=lookup[ref]
        if Path(r['image_url']).name!=name or split['assignments'][ref]['fold']!='test':raise ValueError(f'Wrong held-out image: {ref}')
        queries.append({'id':f'large_image_{number:02}','task':'image','question':question,'target_dataset':'DRISHTIKON',
            'relevant_images':[str(Path(r['image_url']).relative_to(ROOT))],'gold_record_ref':ref,'fold':'test',
            'evaluation_role':'group_heldout_source_verified','source_verified':True,'visual_content_verified':True,
            'human_reviewed':False,'relevance_complete':False,'modality':'image_associated',
            'source_evidence':visual,'label_source':'Downloaded image directly inspected; query describes its visible content'})
    for q in queries:q['evaluation_group']=split['assignments'][q['gold_record_ref']]['group']
    texts=[q for q in queries if q['task']=='text'];images=[q for q in queries if q['task']=='image']
    if len(texts)!=100 or len(images)!=20:raise ValueError('Expected 100 text and 20 image questions.')
    counts=collections.Counter(q['target_dataset'] for q in texts)
    if set(counts.values())!={20}:raise ValueError(f'Unbalanced text cases: {counts}')
    if len({canonical(q['question']) for q in queries})!=120:raise ValueError('Repeated question wording.')
    if len({image_digest(ROOT/q['relevant_images'][0]) for q in images})!=20:raise ValueError('Duplicate image contents.')
    groups={ds:len({q['evaluation_group'] for q in texts if q['target_dataset']==ds}) for ds in counts}
    doc={'description':'100 authored text questions (20 per dataset) and 20 inspected image queries. Verified source positives; labels require human review and are not exhaustive relevance judgements.',
        'split_manifest':'research/split_manifest.json','text_questions_per_dataset':dict(counts),'distinct_text_source_groups':groups,
        'query_expansion_policy':'API expansion receives only each question, never expected answers or evidence. Same cached phrasings are shared by all compared conditions.',
        'sampling_note':'Indian Cultural Dataset has 15 held-out records; five provide a second fact question. Uncertainty estimates must resample source groups rather than treating these questions as independent.',
        'queries':texts+images}
    output=ROOT/'validation_queries_120.json';output.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    (ROOT/'questions_100.txt').write_text('\n'.join(q['question'] for q in texts)+'\n')
    (ROOT/'image_questions_20.txt').write_text('\n'.join(q['question'] for q in images)+'\n')
    with (RESEARCH/'annotation_review_120.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['query_id','question','accepted_answers','positive_evidence','source_evidence','human_reviewed','reviewer','additional_relevant_refs','notes','relevance_complete'])
        for q in doc['queries']:w.writerow([q['id'],q['question'],json.dumps(q.get('expected_answers',[]),ensure_ascii=False),'; '.join(q.get('relevant_records',q.get('relevant_images',[]))),q.get('source_evidence',''),False,'','','',False])
    print('Saved',output,'with',len(texts),'text and',len(images),'image cases.')
    print('Text source groups:',groups)
if __name__=='__main__':build()
