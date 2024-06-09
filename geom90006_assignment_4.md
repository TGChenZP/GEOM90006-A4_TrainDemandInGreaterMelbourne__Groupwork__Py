**Spatial Data Analytics (GEOM90006)**

# Assignment 4: Major Project

## Objectives

1. To undertake, in pairs, a self-guided spatial or spatiotemporal investigation consisting of multiple strands of analysis.
2. To extend knowledge from previous assignments and apply spatial analysis concepts that were not explored in previous assignments.
3. To present spatial information to a professional but non-technical audience – in written and oral forms.

## Your task

Your first job is to identify your project **topic**. Depending on your group's strengths and interests, you may select a topic with an urban or an environmental theme. Examples could include: housing, crime, agriculture, ecology, pollution, ... 

Then decide on your **research question**. This should be quite high-level, rather than technique-specific. What question(s) do you want to answer using spatial analysis as your tool?

As for **data**, your project must include analysis (not just visualisation) of at least one point dataset **and** at least one polygon dataset **and** at least one raster dataset. Some examples:

- **Point data** – traffic crashes, weather stations, dams, buildings, fire stations, schools, intersections, ...
- **Polygon data** – admin boundaries (localities, LGAs, states, ...), planning zones, census data, ...
- **Raster data** – DEM, rainfall, soil type, geology, land use, land cover, imagery, ...

There is no restriction on the geographic area of your project. However, we provide additional support for finding relevant data sources for Melbourne, Victoria or Australia: <https://unimelb.libguides.com/GIS>

A compulsory component of Assignment 4 is **Exploratory Spatial Data Analysis (ESDA)**. This should be the first set of activities related to your chosen topic.

You are expected to:

- Generate tables and/or plots (histograms, scatterplots, ...) that demonstrate essential (non-spatial) statistical properties of the data relating to attributes relevant to your research question.
- Generate a pivot table or pivot chart to summarise some aspect of your data. For example, if you were analysing car crashes in Victoria, you might list the top 10 LGAs in Victoria by number of crashes.
- Generate cartographic-quality map(s) of your data.
- Give a brief description and interpretation of the insights gained from the ESDA.

Then you will proceed to undertake your chosen **analysis**. Here, you have two choices:

- You may wish to start with datasets that were provided in Assignment 2 or 3. In this case, you must extend yourself to analysis concepts and techniques that were not explored in these assignments.
  - These techniques might be those covered in the lectures (such as geographically weighted regression, EBK, and so on), or other techniques found in textbooks or other resources.
  - You may choose to incorporate additional datasets on the same theme if you wish.
  - You are allowed to use analysis techniques from Assignments 2 and 3 as part of a comparison of evaluation of techniques if you wish, but they should form a minor part of your work.
- Alternatively, you may wish to set your own theme (centred around an urban or environmental topic) and source your own datasets, applying analysis techniques already explored in Assignment 2 or 3 or any other techniques of your choice.
  - In this scenario, interpretation is key.

As part of your project, you must conduct **research** and include references where appropriate. 

## Outline of submissions

### *Proposal (5% of subject)*

The project proposal is a two- to three-page document that introduces your project and describes how you intend to complete it.

- Give your project a good, succinct title.

- State your chosen topic and research question.

- State the project's scope, including the geographic area(s), spatial scale and/or time scope as relevant. 

- State the proposed data sources. 

- Present a simple cartographic-quality map that demarcates the project focus area(s) and displays one or two spatial data layers relevant to the topic of the project. **Do not** perform any analysis steps – filtering/clipping is the most you should do at this stage.

- Present a brief literature review of around 300 to 400 words. You might undertake this literature review before settling on your precise research question.

- Set out your proposed methodology at a high level, and describe what types of results you expect to produce. **Do not** perform any analysis steps yet.

- Put forward a hypothesis regarding what you believe your results are likely to show.

- Add a reference list at the end of your document. The usual referencing format in spatial sciences is APA, but if you are more familiar with a different widely-used referencing format, you may use it instead.

**Submission:** Under "proposal", one group member will submit the project proposal as a PDF file, created using any software. The proposal should be two or three pages in length.

### *Formative task (2% of subject)*

The formative task for Assignment 4 involves preparing a small piece of your completed project for early feedback. 

Start a new Jupyter notebook. Place the latest version of your project title and research question at the top.

Then present a map related to one component of your chosen analysis and write an interpretation (around 150 words) of what this map tells you in the context of your research question.

The map may be accompanied by another data visualisation (e.g. chart) if necessary, but the focus is on the map.

**Submission:** Under "formative task", one group member will submit a Jupyter notebook with relevant outputs pre-generated and included in the saved file. There is no need to submit any data, as the markers will not assess or execute your code.

### *Implementation and discussion (35% of subject)*

You will prepare a Jupyter notebook as follows:

- Start with material from your project proposal (project title, topic, research question, scope and data sources). Also write a brief introductory paragraph, which might incorporate key points from your literature review.

- Present the ESDA component of your work as described above under "Your task".

- Then move onto your chosen analysis. 
  - For each analysis, introduce the technique and describe in broad terms the methodology you will apply.
  - As you write code to implement the methodology, show your results and offer interpretations in the context of your research question. Present results as plots, maps, tables, or any other way you see fit.
  - Write one or more discussion sections going into detail about the insights gained into your research question, as well as the techniques used, including strengths, weaknesses and limitations.
  - As part of the University's commitment to sustainability, include recommendations on the issues we are facing in urban and environmental domains regarding *sustainability* (UN Sustainable Development Goals) and *resilience*. 

- Complete your Jupyter notebook by writing a brief overall conclusion.

- Add a reference list at the end of your notebook. The usual referencing format in spatial sciences is APA, but if you are more familiar with a different widely-used referencing format, you may use it instead.

According to the Handbook, the project implementation and discussion is "equivalent to 1500 words" per person. In our view, approximately half of your effort on this task will be spent writing code. Therefore, you are expected to provide around 750 written words per person (around 1500 written words per pair). Note that this is a guide, not a hard limit.

**Submission:** Under "implementation", one group member will submit a **.zip file** consisting of a Jupyter notebook (.ipynb) and **all** required data files. Even the data files we provided to you must be submitted if used. You may use any Python libraries as required - if necessary, provide instructions for the marker on how to obtain these libraries.

### *Video (10% of subject)*

As a pair, you will prepare a 5-minute video summarising your work on Assignment 4.

The video should follow the format:

- 2.5 minutes: Group member 1 introduces the project topic and data, and presents the methodologies
- 2.5 minutes: Group member 2 describes the results and offers a discussion and conclusion

For a group of three, please discuss with your demonstrator.

You must provide a visual component, such as slides (or something more creative), to support your speech. As you speak, your face should be visible somewhere on screen. (There are no exceptions to this. Use a phone or a lab computer at uni if you do not have a camera on your usual computer.)

**Submission:** Under "video", one group member will submit an **.mp4 file** to Canvas.

## Marking criteria

### *Proposal (total 5 marks)*

*All group members receive the same mark for the project proposal.*

- **Research question and scope (1 mark)**   
The research question is clearly formulated, relevant, focused, and capable of being answered. The scope is logical and well chosen.

- **Proposed analysis (2 marks)**   
The proposed analysis is feasible, logical, and addresses the research question. The datasets meet the requirements, are appropriate for addressing the research question, and are described in terms of their source, currency, metadata, geographical extent, etc.

- **Literature review (2 marks)**   
The literature review provides context for, and insight into, the research question. The quality of sources used is suitable for the task. Referencing is done correctly, adhering to APA or some other well-known standard. For full marks, some academic references (to peer-reviewed journal articles) must be included.

### *Formative task (total 2 marks)*

*All group members receive the same mark for the formative task.*

- **Map (1 mark)**
  - 0 marks: The map is difficult to understand. The relevance to the analysis is difficult to grasp.
  - 0.5 marks: The map has some relevance to the analysis but may be somewhat simplistic or can be improved visually. 
  - 1 mark: The map is highly relevant to the analysis and is of professional presentation quality.

- **Discussion (1 mark)**   
  - 0 marks: The discussion is vague and difficult to understand. It barely engages with what is shown in the map.
  - 0.5 marks: The discussion offers some relevant ideas based on the map, but is missing some points or could be argued more convincingly.
  - 1 mark: The discussion directly responds to the map, offering a valid, well-argued analysis and interpretation.

### *Implementation and discussion (total 35 marks)*

*All group members receive the same mark for the project implementation and discussion.*

- **Code: Correctness of output (4 marks)**   
Your Jupyter Notebook is submitted correctly, performs tasks correctly, and produces well-formatted output.

- **Code: Structure, commenting and documentation (4 marks)**   
Your code is well documented and understandable in structure, and in operation of the functions. Code that is modular and re-useable (for instance) will score higher marks. The analytical workflow is clear. Code that is poorly commented will lose marks.

- **ESDA (4 marks)**   
The ESDA techniques are applied correctly to the provided datasets. The results of ESDA are described and interpreted in a suitable way.

- **Own analysis: methodology and results (8 marks)**   
The methodology is made up of a highly relevant workflow or work packages. A deep understanding of the techniques is evident. Results are presented using well-chosen, effective visuals or text outputs. For full marks, there is evidence of creative or innovative thinking in either your methodology design or your results output.

- **Own analysis: interpretation and discussion (8 marks)**   
The discussion offers valid, in-depth, well-argued interpretations of the results. It displays clear insights on the implications and consequences of the results as well as possible limitations and weaknesses. Recommendations are provided, both in general, and in the context of sustainability and resilience.

- **Professional standard (5 marks)**   
Your Jupyter notebook is presented clearly, well structured, interleaving the text with the code to support the transparency and understandability of your analysis. An appropriate introduction and conclusion are provided. The produced maps, graphs and tables are of professional presentation quality. 

- **Research and references (2 marks)**   
The work is informed by research at appropriate points, especially when techniques are introduced or discussed. The quality of sources used is suitable for the task. Referencing is done correctly, adhering to APA or some other well-known standard. For full marks, some academic references (to peer-reviewed journal articles) must be included.

### *Video (total 10 marks)*

- **Individual mark: Content (4 marks)**   
*Each group member is marked individually.*   
The presenter demonstrates a deep level of insight into the data used, the spatial analysis techniques, results, and interpretations (as relevant for Presenter 1 or 2). The research question is introduced at the beginning (Presenter 1) / answered at the end (Presenter 2). The presentation is structured logically and context is provided so that the material can be clearly understood.

- **Individual mark: Expression and style (4 marks)**   
*Each group member is marked individually.*   
The presenter speaks clearly and understandably, varying their tone of voice. The visuals and speech complement each other without duplicating the same material. Visuals are employed effectively to convey the presenter's message. Maps are presented at a professional standard.

- **Group mark: Engagement and creativity of presentation (2 marks)**   
*All group members receive the same mark.*   
The presentation feels like one cohesive presentation, not two separate ones. It is highly engaging to watch and listen to. For full marks, creative and innovative presentation techniques – verbal and/or visual – must be used.
