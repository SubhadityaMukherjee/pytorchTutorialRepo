ran = rand(10,20)

function bnorm_x(x,γ, β, ϵ = 1e-5)
	mean_x = sum(x)/length(x) #mean of x
	variance_x = sum((x .- mean_x).^2)/length(ran) #variance of x
    x̂  = (x .- mean_x)/sqrt(variance_x^2 + ϵ)
    @info size(x̂ ), size(β), size(γ)
    return γ.*x̂  + β 
end

