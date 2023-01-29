from data_util import Coref
coref=Coref()
# text = 'Alice goes down the rabbit hole. Where she would discover a new reality beyond her expectations.'
text="Hipkins served in opposition as Labour's education spokesperson. In the Sixth Labour Government, he previously served as minister of education, police, the public service, and leader of the House. He became a prominent figure as a result of the COVID-19 pandemic in New Zealand, taking on the roles of minister of health from July to November 2020 and minister for COVID-19 response from November 2020 to June 2022."
res=coref.get_resolved_text(text)

print(res)