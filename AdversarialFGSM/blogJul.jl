using Images, ImageView
using Flux

img = Images.load("turtle.jpg")

ImageView.imshow(img)

new_img = channelview(colorview(RGB, img))

function signtest(x)
        if(x == 0)
                return 0
        elseif(x < 0)
                return -1
        else
                return 1
        end
end

function fgsm_attack(image, ϵ, data_grad)
        sign_data_grad = signtest.(data_grad)
        peturbed_img = image + ϵ*sign_data_grad
        peturbed_img = clamp.(peturbed_img, 0,1)
        return peturbed_img
end

imgradient = diff(new_img, dims = 3)

attacked = fgsm_attack(new_img[:, :, 1:end-1], 0.1,imgradient)

ImageView.imshow(permutedims(attacked, (2,3,1)))

