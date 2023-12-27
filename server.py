import sys
import grpc
from concurrent import futures
from proto import embedding_pb2 as embedding_pb2
from proto import embedding_pb2_grpc as embedding_pb2_grpc
from src.model.model import EmbeddingModel

class EmbeddingService(embedding_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self):
        self.models = {}
        self._initialize_model()

    def _initialize_model(self):
        self.models["bert"] = EmbeddingModel(text_model_name='bert', image_model_name='resnet')
        self.models["longformer"] = EmbeddingModel(text_model_name='longformer', image_model_name='resnet')

    def GetImageEmbedding(self, request, context):
        image_path = request.image_path
        if not image_path:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Image path can't be empty.")
        model_name = request.model
        if model_name not in self.models:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "Invalid model name.")

        embedding = self.models[model_name].get_image_embedding(image_path=image_path)
        if embedding is not None:
            return embedding_pb2.EmbeddingResponse(embedding=embedding)
        else:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to generate image embedding.")

    def GetTextEmbedding(self, request, context):
        startTime = time.time()
        input_text = request.input_text
        model_name = request.model

        if not input_text:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Input can't be empty.")

        if model_name not in self.models:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "Invalid model name.")

        embedding = self.models[model_name].get_text_embedding(input_text=input_text)
        total_tokens = self.models[model_name].token_count(input_text=input_text)

        endTime = time.time()
        executionTime = endTime - startTime

        if embedding is not None:
            data = embedding_pb2.Data()
            data.embedding.extend(embedding)  # Ensure embedding is iterable (e.g., a list)
            data.length = len(embedding)

            usage = embedding_pb2.Usage()
            usage.total_tokens = total_tokens

            return embedding_pb2.EmbeddingResponse(data=data,
                                                   model=model_name,
                                                   execution_time=executionTime,
                                                   usage=usage)
        else:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to generate text embedding.")

    def _abort(self, context, code, details):
        context.abort(code, details)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started.")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Server stopped by user.")

if __name__ == '__main__':
    serve()
