from app.graph.workflow import graph

result = graph.invoke({
    "query": "patches in vision transformers"
})

print(result["report"])