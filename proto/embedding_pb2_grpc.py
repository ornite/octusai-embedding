# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto.embedding_pb2 as embedding__pb2


class EmbeddingServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetImageEmbedding = channel.unary_unary(
                '/proto.EmbeddingService/GetImageEmbedding',
                request_serializer=embedding__pb2.ImageRequest.SerializeToString,
                response_deserializer=embedding__pb2.EmbeddingResponse.FromString,
                )
        self.GetTextEmbedding = channel.unary_unary(
                '/proto.EmbeddingService/GetTextEmbedding',
                request_serializer=embedding__pb2.TextRequest.SerializeToString,
                response_deserializer=embedding__pb2.EmbeddingResponse.FromString,
                )


class EmbeddingServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetImageEmbedding(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTextEmbedding(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EmbeddingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetImageEmbedding': grpc.unary_unary_rpc_method_handler(
                    servicer.GetImageEmbedding,
                    request_deserializer=embedding__pb2.ImageRequest.FromString,
                    response_serializer=embedding__pb2.EmbeddingResponse.SerializeToString,
            ),
            'GetTextEmbedding': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTextEmbedding,
                    request_deserializer=embedding__pb2.TextRequest.FromString,
                    response_serializer=embedding__pb2.EmbeddingResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'proto.EmbeddingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EmbeddingService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetImageEmbedding(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.EmbeddingService/GetImageEmbedding',
            embedding__pb2.ImageRequest.SerializeToString,
            embedding__pb2.EmbeddingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTextEmbedding(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.EmbeddingService/GetTextEmbedding',
            embedding__pb2.TextRequest.SerializeToString,
            embedding__pb2.EmbeddingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
