(function() {
    var implementors = Object.fromEntries([["mistralrs_core",[["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.DiffusionLoaderType.html\" title=\"enum mistralrs_core::DiffusionLoaderType\">DiffusionLoaderType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.ImageGenerationResponseFormat.html\" title=\"enum mistralrs_core::ImageGenerationResponseFormat\">ImageGenerationResponseFormat</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.MemoryGpuConfig.html\" title=\"enum mistralrs_core::MemoryGpuConfig\">MemoryGpuConfig</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.ModelDType.html\" title=\"enum mistralrs_core::ModelDType\">ModelDType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.NormalLoaderType.html\" title=\"enum mistralrs_core::NormalLoaderType\">NormalLoaderType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.ToolCallType.html\" title=\"enum mistralrs_core::ToolCallType\">ToolCallType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.VisionLoaderType.html\" title=\"enum mistralrs_core::VisionLoaderType\">VisionLoaderType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"enum\" href=\"mistralrs_core/enum.WebSearchUserLocation.html\" title=\"enum mistralrs_core::WebSearchUserLocation\">WebSearchUserLocation</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ApproximateUserLocation.html\" title=\"struct mistralrs_core::ApproximateUserLocation\">ApproximateUserLocation</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.CalledFunction.html\" title=\"struct mistralrs_core::CalledFunction\">CalledFunction</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ChatCompletionChunkResponse.html\" title=\"struct mistralrs_core::ChatCompletionChunkResponse\">ChatCompletionChunkResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ChatCompletionResponse.html\" title=\"struct mistralrs_core::ChatCompletionResponse\">ChatCompletionResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.Choice.html\" title=\"struct mistralrs_core::Choice\">Choice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ChunkChoice.html\" title=\"struct mistralrs_core::ChunkChoice\">ChunkChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.CompletionChoice.html\" title=\"struct mistralrs_core::CompletionChoice\">CompletionChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.CompletionChunkChoice.html\" title=\"struct mistralrs_core::CompletionChunkChoice\">CompletionChunkChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.CompletionChunkResponse.html\" title=\"struct mistralrs_core::CompletionChunkResponse\">CompletionChunkResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.CompletionResponse.html\" title=\"struct mistralrs_core::CompletionResponse\">CompletionResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.Delta.html\" title=\"struct mistralrs_core::Delta\">Delta</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.DiffusionGenerationParams.html\" title=\"struct mistralrs_core::DiffusionGenerationParams\">DiffusionGenerationParams</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ImageChoice.html\" title=\"struct mistralrs_core::ImageChoice\">ImageChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ImageGenerationResponse.html\" title=\"struct mistralrs_core::ImageGenerationResponse\">ImageGenerationResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.Logprobs.html\" title=\"struct mistralrs_core::Logprobs\">Logprobs</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ResponseLogprob.html\" title=\"struct mistralrs_core::ResponseLogprob\">ResponseLogprob</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ResponseMessage.html\" title=\"struct mistralrs_core::ResponseMessage\">ResponseMessage</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.ToolCallResponse.html\" title=\"struct mistralrs_core::ToolCallResponse\">ToolCallResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.TopLogprob.html\" title=\"struct mistralrs_core::TopLogprob\">TopLogprob</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.Usage.html\" title=\"struct mistralrs_core::Usage\">Usage</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a <a class=\"struct\" href=\"mistralrs_core/struct.WebSearchOptions.html\" title=\"struct mistralrs_core::WebSearchOptions\">WebSearchOptions</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"enum\" href=\"mistralrs_core/enum.DiffusionLoaderType.html\" title=\"enum mistralrs_core::DiffusionLoaderType\">DiffusionLoaderType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"enum\" href=\"mistralrs_core/enum.ImageGenerationResponseFormat.html\" title=\"enum mistralrs_core::ImageGenerationResponseFormat\">ImageGenerationResponseFormat</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"enum\" href=\"mistralrs_core/enum.ModelDType.html\" title=\"enum mistralrs_core::ModelDType\">ModelDType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"enum\" href=\"mistralrs_core/enum.NormalLoaderType.html\" title=\"enum mistralrs_core::NormalLoaderType\">NormalLoaderType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"enum\" href=\"mistralrs_core/enum.ToolCallType.html\" title=\"enum mistralrs_core::ToolCallType\">ToolCallType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"enum\" href=\"mistralrs_core/enum.VisionLoaderType.html\" title=\"enum mistralrs_core::VisionLoaderType\">VisionLoaderType</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ApproximateUserLocation.html\" title=\"struct mistralrs_core::ApproximateUserLocation\">ApproximateUserLocation</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.CalledFunction.html\" title=\"struct mistralrs_core::CalledFunction\">CalledFunction</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ChatCompletionChunkResponse.html\" title=\"struct mistralrs_core::ChatCompletionChunkResponse\">ChatCompletionChunkResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ChatCompletionResponse.html\" title=\"struct mistralrs_core::ChatCompletionResponse\">ChatCompletionResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.Choice.html\" title=\"struct mistralrs_core::Choice\">Choice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ChunkChoice.html\" title=\"struct mistralrs_core::ChunkChoice\">ChunkChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.CompletionChoice.html\" title=\"struct mistralrs_core::CompletionChoice\">CompletionChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.CompletionChunkChoice.html\" title=\"struct mistralrs_core::CompletionChunkChoice\">CompletionChunkChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.CompletionChunkResponse.html\" title=\"struct mistralrs_core::CompletionChunkResponse\">CompletionChunkResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.CompletionResponse.html\" title=\"struct mistralrs_core::CompletionResponse\">CompletionResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.Delta.html\" title=\"struct mistralrs_core::Delta\">Delta</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.DiffusionGenerationParams.html\" title=\"struct mistralrs_core::DiffusionGenerationParams\">DiffusionGenerationParams</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ImageChoice.html\" title=\"struct mistralrs_core::ImageChoice\">ImageChoice</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ImageGenerationResponse.html\" title=\"struct mistralrs_core::ImageGenerationResponse\">ImageGenerationResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.Logprobs.html\" title=\"struct mistralrs_core::Logprobs\">Logprobs</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ResponseLogprob.html\" title=\"struct mistralrs_core::ResponseLogprob\">ResponseLogprob</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ResponseMessage.html\" title=\"struct mistralrs_core::ResponseMessage\">ResponseMessage</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.ToolCallResponse.html\" title=\"struct mistralrs_core::ToolCallResponse\">ToolCallResponse</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.TopLogprob.html\" title=\"struct mistralrs_core::TopLogprob\">TopLogprob</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.Usage.html\" title=\"struct mistralrs_core::Usage\">Usage</a>"],["impl&lt;'a, 'py&gt; PyFunctionArgument&lt;'a, 'py&gt; for &amp;'a mut <a class=\"struct\" href=\"mistralrs_core/struct.WebSearchOptions.html\" title=\"struct mistralrs_core::WebSearchOptions\">WebSearchOptions</a>"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[12040]}