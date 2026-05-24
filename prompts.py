DECOMPOSER_SYSTEM_PROMPT = (
    "You are Decomposer. Given an input function spec, return only the output "
    "function body. Write one task-specific Python decomposition for answering the "
    "question with context_qa_model. You are good at decomposing complex "
    "questions into simpler, explicit subquestions. context_qa_model is more "
    "reliable on simple, unambiguous, context-grounded questions than on long "
    "or compositionally complex questions. It returns a short answer string "
    "when the question is unambiguously answerable from context and raises "
    "ValueError otherwise. Do not catch ValueError and do not write fallback "
    "logic; an external controller may fall back to the baseline. Decompose "
    "only when the subquestions are safer and better-defined than the original "
    "question. Do not invent hidden relations, drop restrictive modifiers, "
    "create ambiguous intermediate entities, or assume facts not implied by "
    "the question. If decomposition would be unsafe, make the decomposition a "
    "less-decomposed constrained question or the original question. Ask the "
    "final question for the exact answer type requested by the original "
    "question. Use only Python built-ins, the function parameters, and "
    "context_qa_model. Start the first body line immediately without leading "
    "spaces. After each newline, indent top-level body lines by 4 spaces, and "
    "use 4 additional spaces for each nested block. Do not include the def line "
    "or docstring. Do not use Markdown, labels, imports, or helper function "
    "definitions."
)


def build_func_spec(question: str) -> str:
    return f'''def main(context, context_qa_model):
    """Answer the question: {question}

    Args:
        context (str): A context that contains the necessary information to
            answer a question.
        context_qa_model (Callable[[str, str], str]): Function that
            takes two arguments:
            - question (str): Any question answerable from context.
            - context (str): The same formatted MuSiQue context string passed
              to main.
            It returns a short answer string grounded in the context. It raises
            ValueError if the question is not unambiguously answerable from
            context.

    Returns:
        str: The final answer to the question.
    """'''


DECOMPOSER_EXAMPLES = [
    {
        "func_spec": build_func_spec(
            "When does monsoon season occur in the city where the torch relay "
            "happened in India?"
        ),
        "func_body": """city = context_qa_model('In which city did the torch relay happen in India?', context)
    return context_qa_model(f'When does monsoon season occur in {city}?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "When did the publisher of Lunicus announce the remastered release "
            "of star trek for tv?"
        ),
        "func_body": """return context_qa_model('When was the remastered release of star trek for tv announced?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "who was the spouse of the politician that wrote the majority of "
            "the federalist papers?"
        ),
        "func_body": """politician = context_qa_model('What politician is mentioned in the context as having written the majority of the federalist papers?', context)
    return context_qa_model(f'Who was the spouse of {politician}?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "Who was sent to the country the performer of Je vais me marier, "
            "Marie belongs to?"
        ),
        "func_body": """performer = context_qa_model('Who was the performer of "Je vais me marier, Marie"?', context)
    country = context_qa_model(f'Which country does {performer} belong to?', context)
    return context_qa_model(f'Who was mentioned in the context as having been sent to {country}?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "Where does the columbia river meet where the warm moist air mass "
            "over the andes mountains in the country of Amanda O come from?"
        ),
        "func_body": """source = context_qa_model('Where does the warm moist air mass over the Andes mountains come from?', context)
    return context_qa_model(f'Where does the Columbia river meet {source}?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "What's the citizen country of the actor who plays John, in the "
            "movie where the title requests a meeting, in the city where James "
            "Cuno was born?"
        ),
        "func_body": """city = context_qa_model('In which city was James Cuno born?', context)
    movie = context_qa_model(f'The title of what movie requests a meeting in {city}?', context)
    actor = context_qa_model(f'Which actor plays John in {movie}?', context)
    return context_qa_model(f"What's the citizen country of {actor}?", context)""",
    },
    {
        "func_spec": build_func_spec(
            "What was the 2018 population of the host of the 1920 Summer "
            "Olympics, that involved the country that has a spiral viaduct in "
            "the birthplace of Karin Thomas?"
        ),
        "func_body": """host = context_qa_model('What host did 1920 Summer Olympics have?', context)
    return context_qa_model(f'What was the 2018 population of {host}?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "What is the county having the city containing the National "
            "Historic Site of the president of the state housing Tana Glacier "
            "when the state was purchased?"
        ),
        "func_body": """state = context_qa_model('What state is mentioned in the context as housing Tana Glacier?', context)
    president = context_qa_model(f'Who was president when {state} was purchased?', context)
    city = context_qa_model(f'Which city contains the National Historic Site of {president}?', context)
    return context_qa_model(f'What county contains {city}?', context)""",
    },
    {
        "func_spec": build_func_spec(
            "Where is the lowest place in the country which, along with "
            "Eisenhower's VP's country, recognized Gaddafi's government early "
            "on?"
        ),
        "func_body": """vice_president = context_qa_model("Who served as Eisenhower's vice president?", context)
    country = context_qa_model(f"Which country, along with {vice_president}'s country, recognized Gaddafi's government at an early date?", context)
    return context_qa_model(f'Where is the lowest place in {country}?', context)""",
    },
]


DECOMPOSER_FEW_SHOT_PROMPT = "\n\n---\n\n".join(
    f"{example['func_spec']}\n{example['func_body']}"
    for example in DECOMPOSER_EXAMPLES
)


def build_decomposer_messages(func_spec: str) -> list[dict]:
    messages = [{"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT}]
    for example in DECOMPOSER_EXAMPLES:
        messages.append({"role": "user", "content": example["func_spec"]})
        messages.append({"role": "assistant", "content": example["func_body"]})
    messages.append({"role": "user", "content": func_spec})
    return messages


CONTEXT_QA_SYSTEM_PROMPT = (
    "You answer questions using only the provided context. If the question has "
    "exactly one short answer that is unambiguously supported by the context, "
    "return only that answer string. Do not explain the answer. If the answer "
    "is missing, ambiguous, or not uniquely determined by the context, "
    "return exactly \"Unanswerable\"."
)


def format_qa_user_content(question: str, context: str) -> str:
    return f"""Context:
{context}

Question: {question}"""


CONTEXT_QA_EXAMPLES = [
    {
        "question": "Who was the mayor of New Delhi during the torch relay?",
        "context": """[1] 2008 Summer Olympics torch relay
North Korea: The event was held in Pyongyang on April 28. It was the first time that the Olympic torch has traveled to North Korea. A crowd of thousands waving pink paper flowers and small flags with the Beijing Olympics logo were organized by the authoritarian regime watched the beginning of the relay in Pyongyang, some waving Chinese flags. The event was presided over by the head of the country's parliament, Kim Yong Nam. The North, an ally of China, has been critical of disruptions to the torch relay elsewhere and has supported Beijing in its actions against protests in Tibet. Kim passed the torch to the first runner Pak Du Ik, who played on North Korea's 1966 World Cup soccer team, as he began the 19-kilometre route through Pyongyang. The relay began from the large sculpted flame of the obelisk of the Juche Tower, which commemorates the national ideology of Juche, or "self-reliance", created by the country's late founding President Kim Il Sung, father of leader Kim Jong Il, who did not attend.

[2] 2008 Summer Olympics torch relay
The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of "one world, one dream". Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the "Journey of Harmony", lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics.

[3] 2018 Winter Olympics torch relay
The 2018 Winter Olympics torch relay began 24 October 2017 and ended on 9 February 2018, in advance of the 2018 Winter Olympics. After being lit in Olympia, Greece, the torch traveled to Athens on 31 October. The torch began its Korean journey on 1 November, visiting all Regions of Korea. The Korean leg began in Incheon: the torch travelled across the country for 101 days. 7,500 relay runners participated in the torch relay over a distance of 2,017 km. The torchbearers each carried the flame for 200 metres. The relay ended in Pyeongchang's Olympic Stadium, the main venue of the 2018 Olympics. The final torch was lit by figure skater Yuna Kim.

[4] New Delhi
The climate of New Delhi is a monsoon-influenced humid subtropical climate (Köppen Cwa) with high variation between summer and winter in terms of both temperature and rainfall. The temperature varies from 46 °C (115 °F) in summers to around 0 °C (32 °F) in winters. The area's version of a humid subtropical climate is noticeably different from many other cities with this climate classification in that it features long and very hot summers, relatively dry and mild winters, a monsoonal period, and dust storms. Summers are long, extending from early April to October, with the monsoon season occurring in the middle of the summer. Winter starts in November and peaks in January. The annual mean temperature is around 25 °C (77 °F); monthly daily mean temperatures range from approximately 14 to 34 °C (57 to 93 °F). New Delhi's highest temperature ever recorded is 49.1 °C (120.4 °F) while the lowest temperature ever recorded is −3.2 °C (26.2 °F). Those for Delhi metropolis stand at 49.9 °C (121.8 °F) and −3.2 °C (26.2 °F) respectively. The average annual rainfall is 784 millimetres (30.9 in), most of which is during the monsoons in July and August.

[5] North American Monsoon
The North American monsoon, variously known as the Southwest monsoon, the Mexican monsoon, the New Mexican monsoon, or the Arizona monsoon, is a pattern of pronounced increase in thunderstorms and rainfall over large areas of the southwestern United States and northwestern Mexico, typically occurring between July and mid September. During the monsoon, thunderstorms are fueled by daytime heating and build up during the late afternoon - early evening. Typically, these storms dissipate by late night, and the next day starts out fair, with the cycle repeating daily. The monsoon typically loses its energy by mid-September when drier and cooler conditions are reestablished over the region. Geographically, the North American monsoon precipitation region is centered over the Sierra Madre Occidental in the Mexican states of Sinaloa, Durango, Sonora and Chihuahua.

[6] Climate of India
The Climate of India comprises a wide range of weather conditions across a vast geographic scale and varied topography, making generalisations difficult. Based on the Köppen system, India hosts six major climatic subtypes, ranging from arid desert in the west, alpine tundra and glaciers in the north, and humid tropical regions supporting rainforests in the southwest and the island territories. Many regions have starkly different microclimates. The country's meteorological department follows the international standard of four climatological seasons with some local adjustments: winter (December, January and February), summer (March, April and May), a monsoon rainy season (June to September), and a post-monsoon period (October to November).

[7] Climate of Pakistan
Western Disturbances mostly occur during the winter months and cause light to moderate showers in southern parts of the country while moderate to heavy showers with heavy snowfall in the northern parts of the country. These westerly waves are robbed of most of the moisture by the time they reach Pakistan. Fog occurs during the winter season and remains for weeks in upper Sindh, central Khyber Pakhtunkhwa and Punjab. Southwest Monsoon occurs in summer from the month of June till September in almost whole Pakistan excluding western Balochistan, FATA, Chitral and Gilgit -- Baltistan. Monsoon rains bring much awaited relief from the scorching summer heat. These monsoon rains are quite heavy by nature and can cause significant flooding, even severe flooding if they interact with westerly waves in the upper parts of the country. Tropical Storms usually form during the summer months from late April till June and then from late September till November. They affect the coastal localities of the country. Dust storms occur during summer months with peak in May and June, They are locally known as Andhi. These dust storms are quite violent. Dust storms during the early summer indicate the arrival of the monsoons while dust storms in the autumn indicate the arrival of winter. Heat waves occur during May and June, especially in southern Punjab, central Balochistan and interior Sindh. Thunderstorms most commonly occur in northern Punjab, Khyber Pakhtunkhwa and Azad Kashmir. Continental air prevails during the period when there is no precipitation in the country.

[8] 2008 Summer Olympics torch relay
After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch traveled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event.

[9] 2008 Summer Olympics torch relay
India: Due to concerns about pro-Tibet protests, the relay through New Delhi on April 17 was cut to just 2.3 km (less than 1.5 miles), which was shared amongst 70 runners. It concluded at the India Gate. The event was peaceful due to the public not being allowed at the relay. A total of five intended torchbearers -Kiran Bedi, Soha Ali Khan, Sachin Tendulkar, Bhaichung Bhutia and Sunil Gavaskar- withdrew from the event, citing "personal reasons", or, in Bhutia's case, explicitly wishing to "stand by the people of Tibet and their struggle" and protest against the PRC "crackdown" in Tibet. Indian national football captain, Baichung Bhutia refused to take part in the Indian leg of the torch relay, citing concerns over Tibet. Bhutia, who is Sikkimese, is the first athlete to refuse to run with the torch. Indian film star Aamir Khan states on his personal blog that the "Olympic Games do not belong to China" and confirms taking part in the torch relay "with a prayer in his heart for the people of Tibet, and ... for all people across the world who are victims of human rights violations". Rahul Gandhi, son of the Congress President Sonia Gandhi and scion of the Nehru-Gandhi family, also refused to carry the torch.

[10] 2008 Summer Olympics torch relay
Thailand: The April 18 relay through Bangkok was the Olympic flame's first visit to Thailand. The relay covered just over 10 km, and included Bangkok's Chinatown. The torch was carried past Democracy Monument, Chitralada Palace and a number of other city landmarks. M.R. Narisa Chakrabongse, Green World Foundation (GWF) chairwoman, withdrew from the torch-running ceremony, protesting against China's actions in Tibet. Several hundred protesters were present, along with Olympic supporters. Thai authorities threatened to arrest foreign protesters and ban them from future entry into Thailand. A coalition of Thai human rights groups announced that it would organise a "small demonstration" during the relay, and several hundred people did indeed take part in protests, facing Beijing supporters. Intended torchbearer Mom Rajawongse Narissara Chakrabongse boycotted the relay, to protest against China's actions in Tibet. In Bangkok, students told the media that the Chinese Embassy provided them with transportation and gave them shirts to wear.

[11] 2008 Summer Olympics torch relay
Argentina: The torch relay leg in Buenos Aires, Argentina, held on April 11, began with an artistic show at the Lola Mora amphitheatre in Costanera Sur. In the end of the show the mayor of Buenos Aires Mauricio Macri gave the torch to the first torchbearer, Carlos Espínola. The leg finished at the Buenos Aires Riding Club in the Palermo district, the last torchbearer being Gabriela Sabatini. The 13.8 km route included landmarks like the obelisk and Plaza de Mayo. The day was marked by several pro-Tibet protests, which included a giant banner reading "Free Tibet", and an alternative "human rights torch" that was lit by protesters and paraded along the route the flame was to take. Most of these protests were peaceful in nature, and the torch was not impeded. Chinese immigrants also turned out in support of the Games, but only minor scuffles were reported between both groups. Runners surrounded by rows of security carried the Olympic flame past thousands of jubilant Argentines in the most trouble-free torch relay in nearly a week. People showered the parade route with confetti as banks, government offices and businesses took an impromptu half-day holiday for the only Latin American stop on the flame's five-continent journey.""",
        "answer": "Unanswerable",
    },
    {
        "question": "What is the county having the city containing the National Historic Site of the president of the state housing Tana Glacier when the state was purchased?",
        "context": """[1] Monona County Courthouse
The Monona County Courthouse, located in Onawa, Iowa, United States, was built in 1892. It was listed on the National Register of Historic Places in 1981 as a part of the County Courthouses in Iowa Thematic Resource. The courthouse is the third building the county has used for court functions and county administration.

[2] Andrew Johnson National Cemetery
The Andrew Johnson National Cemetery is a United States National Cemetery on the grounds of the Andrew Johnson National Historic Site in Greeneville, Tennessee. Established in 1906, the cemetery was built around the resting place of Andrew Johnson, the 17th President of the United States, and holds more than two thousand graves.

[3] Territories of the United States
Territories of the United States are sub-national administrative divisions directly overseen by the United States Federal Government. Unlike U.S. states and Native American tribes which exercise limited sovereignty alongside the federal government, territories are without sovereignty. The territories are classified by whether they are incorporated and whether they have an ``organized ''government through an Organic Act passed by the U.S. Congress.

[4] Eisenhower National Historic Site
Eisenhower National Historic Site preserves the home and farm of Dwight D. Eisenhower, the 34th President of the United States, and its surrounding property of . It is located in Cumberland Township, Adams County, Pennsylvania, just outside Gettysburg. Purchased by then-General Eisenhower and his wife Mamie in 1950, the farm served as a weekend retreat for the President and a meeting place for world leaders, and became the Eisenhowers' home after they left the White House in 1961.

[5] Greeneville, Tennessee
Greeneville is a town in, and the county seat of Greene County, Tennessee, United States. The population as of the 2010 census was 15,062. The town was named in honor of Revolutionary War hero Nathanael Greene. It is the only town with this spelling in the United States, although there are numerous U.S. towns named "Greenville". The town was the capital of the short-lived State of Franklin in the 18th-century history of the Tennessee region.

[6] Ulysses S. Grant National Historic Site
Ulysses S. Grant National Historic Site is a United States National Historic Site located 10 miles (16 km) southwest of Downtown St. Louis, Missouri within the municipality of Grantwood Village. The site, also known as White Haven, commemorates the life, military career, and Presidency of Ulysses S. Grant. Five historic structures are preserved at the site including the childhood home of Julia Dent Grant, wife of Ulysses S. Grant.

[7] Union territory
A union territory is a type of administrative division in the Republic of India. Unlike states, which have their own elected governments, union territories are ruled directly by the Union Government (central government), hence the name ``union territory ''. Union territories in India qualify as federal territories, by definition.

[8] Alaska Purchase
The Alaska Purchase (Russian: Продажа Аляски, tr. Prodazha Alyaski) was the United States' acquisition of Alaska from the Russian Empire on March 30, 1867, by a treaty ratified by the United States Senate, and signed by president Andrew Johnson.

[9] Puʻukoholā Heiau National Historic Site
Puukoholā Heiau National Historic Site is a United States National Historic Site located on the northwestern coast of the island of Hawaii. The site preserves the National Historic Landmark ruins of the last major Ancient Hawaiian temple, and other historic sites.

[10] States of Germany
Local associations of a special kind are an amalgamation of one or more Landkreise with one or more Kreisfreie Städte to form a replacement of the aforementioned administrative entities at the district level. They are intended to implement simplification of administration at that level. Typically, a district-free city or town and its urban hinterland are grouped into such an association, or Kommunalverband besonderer Art. Such an organization requires the issuing of special laws by the governing state, since they are not covered by the normal administrative structure of the respective states.

[11] Tana Glacier
Tana Glacier is a 17-mile-long (27 km) glacier in the U.S. state of Alaska. It begins at Bagley Icefield and flows northwest to its 1950 terminus near the head of the Tana River. Its name, of Alaska Native origin, was first recorded by prospectors in 1900.

[12] Khabarovsky District
Khabarovsky District () is an administrative and municipal district (raion), one of the seventeen in Khabarovsk Krai, Russia. It consists of two unconnected segments separated by the territory of Amursky District, which are located in the southwest of the krai. The area of the district is . Its administrative center is the city of Khabarovsk (which is not administratively a part of the district). Population:""",
        "answer": "Greene County",
    },
]


def build_context_qa_messages(question: str, context: str) -> list[dict]:
    messages = [
        {"role": "system", "content": CONTEXT_QA_SYSTEM_PROMPT},
    ]
    for example in CONTEXT_QA_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": format_qa_user_content(
                    example["question"],
                    example["context"],
                ),
            }
        )
        messages.append({"role": "assistant", "content": example["answer"]})
    messages.append(
        {"role": "user", "content": format_qa_user_content(question, context)}
    )
    return messages


build_baseline_messages = build_context_qa_messages
