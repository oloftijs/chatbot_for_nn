from ranking import full_similarity_search, get_prompt, vectordb, llm, process_file
from train import load
from llama_index import ServiceContext


def main():
    service_context = ServiceContext.from_defaults(llm=llm)
    index = load()
    print("hello, i am a chatbot. Ask me anything!")
    chat_engine = index.as_chat_engine(service_context=service_context)
    while True:
        user_input=input()
        context = full_similarity_search(vectordb=vectordb, input=user_input, k=5)
        prompt = get_prompt(user_input, context)
        response = chat_engine.chat(prompt)
        print(response)
        sources = ""
        sources_list = []
        for i in range(len(context)):
            if context[i].metadata not in sources_list:
                sources += process_file(context[i].metadata) 
                sources_list.append(context[i].metadata)
        print("sources: " + sources)
        print("is there something else i can help you with?")

main()