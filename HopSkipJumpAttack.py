import torch
from torchattacks.attack import Attack
# from ..attack import Attack

from tqdm import tqdm


class HSJA(Attack):
    r"""
    "HopSkipJumpAttack" in the paper 'HopSkipJumpAttack: A Query-Efficient Decision-Based Attack'
    Paper's source: [https://arxiv.org/abs/1904.02144]
    Modified from source code: [https://github.com/Jianbo-Lab/HSJA]
    Distance Measure : L2, Linf
    Arguments:
        model (nn.Module): model to attack.
        clip_max (int): upper bound of the image.
        clip_min (int): lower bound of the image.
        norm (string): Lp-norm to minimize. ['L2', 'Linf'] (Default: 'L2')
        num_iterations (int): number of iterations. (Default: 20)
        gamma (float): used to set binary search threshold theta. The binary search
	      threshold theta is gamma / d^{3/2} for L2 attack and gamma / d^2 for
	      Linf attack. (Default: 1.0)

        Targeted mode(included two parameters): {target_label(int): None for nontargeted attack. (Default: None), 
        target_image(torch.Tensor): an tensor(torch) with the same size as x, or None. (Default: None)}
        
        stepsize_search(str): choose between 'geometric_progression', 'grid_search'. (Default: geometric_progression)
        max_num_evals(int): maximum number of evaluations for estimating gradient (for each iteration).
	    init_num_evals(int): initial number of evaluations for estimating gradient.


    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        # >>> attack = torchattacks.HSJA(model, model, clip_max=1, clip_min=0, norm='L2', num_iterations=20, gamma=1.0,
                 target_label=None, target_image=None, stepsize_search='geometric_progression', max_num_evals=10000,
                 init_num_evals=100, verbose=True)
        # >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, clip_max=1, clip_min=0, norm='L2', num_iterations=20, gamma=1.0,
                 target_label=None, target_image=None, stepsize_search='geometric_progression', max_num_evals=10000,
                 init_num_evals=100, verbose=True):
        super().__init__("HSJA", model)
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.norm = norm
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.target_image = target_image
        self.stepsize_search = stepsize_search
        self.max_num_evals = max_num_evals
        self.init_num_evals = init_num_evals
        self.verbose = verbose
        if target_label:
            self.target_label = torch.tensor(target_label).to(self.device)
        else:
            self.target_label = target_label

    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = []
        if len(labels.shape) == 0:
            adv_images = self.perturb(images, labels)
        else:
            for idx in range(len(images)):
                print("The %d-th adversarial sample is being generated."%(idx + 1))
                temp = self.perturb(images[idx], labels[idx])
                adv_images.append(temp.tolist())
            adv_images = torch.tensor(adv_images)
        return adv_images
    
    
    def decision_function(self, image, labels):
        """
	      Decision function output 1 on the desired side of the boundary,
	      0 otherwise.
	      """
        
        images = self.clip_image(image, self.clip_min, self.clip_max)
        prob = self.get_logits(images)
        if self.target_label is None:
            return torch.argmax(prob, dim=1) != labels
        else:
            return torch.argmax(prob, dim=1) == self.target_label
        

    def clip_image(self, images, clip_min, clip_max):
        """
        Clip an image, or an image batch, with upper and lower threshold.
        """
        
        tmp_tensor = torch.clip(images, min= clip_min, max=clip_max)
        if len(tmp_tensor.shape) == 5 and tmp_tensor.shape[0] == 1:
            return tmp_tensor.squeeze(0).to(self.device)
        else:
            return torch.clip(images, min= clip_min, max=clip_max)

    def compute_distance(self, x_origin, x_perturb):
        """
        Compute the distance between two images.
        """

        if self.norm == 'L2':
            return torch.linalg.norm(x_origin - x_perturb)
        elif self.norm == 'Linf':
            return torch.max(abs(x_origin - x_perturb))


    def approximate_gradient(self, x, num_evals, delta, labels):
        """
        Generate random vectors.
        """
        
        noise_shape = [num_evals] + list(x.shape)
        if self.norm == 'L2':
            rv = torch.randn(*noise_shape)
        elif self.norm == 'Linf':
            rv = (2 * torch.rand(noise_shape) - 1)
        
        rv = rv / torch.sqrt(torch.sum(rv ** 2, (1, 2, 3), keepdims=True)) ##?
        rv = rv.to(self.device)
        y = x + delta * rv
        y = self.clip_image(y, self.clip_min, self.clip_max)
        rv = (y - x) / delta

        decisions = self.decision_function(y, labels)
        decision_shape = [len(decisions)] + [1] * len(x.shape) ## ?
        fval = 2 * decisions.type(torch.float64).reshape(decision_shape) - 1.0 ## ?

        """
        Baseline subtraction (when fval differs), when = 1.0: label changes,
        when = -1.0: label not change
        """

        if torch.mean(fval) == 1.0:
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval) == -1.0:
            gradf = - torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)

        # Get the gradient direction.
        gradf = gradf / torch.linalg.norm(gradf)
        return gradf

    def project(self, x, y, alphas):
        alphas_shape = [len(alphas)] + [1] * len(x.shape)
        alphas = alphas.reshape(alphas_shape).to(self.device)
        if self.norm == 'L2':
            return (1 - alphas) * x + alphas * y
        elif self.norm == 'Linf':
            out_images = self.clip_image(
                y,
                x - alphas,
                x + alphas
            )
        return out_images

    def define_theta(self, x):
        Total_num = torch.tensor(x.numel())
        if self.norm == 'L2':
            theta = self.gamma / ((torch.sqrt(Total_num)) * (Total_num))
        else:
            theta = self.gamma / ((Total_num) ** 2.0)
        return theta

    def binary_search_batch(self, x, y, labels):
        """ Binary search to approach the boundar. """
        
        theta = self.define_theta(x)
        dists_post_update = torch.tensor([
            self.compute_distance(
                x,
                perturbed_image,
            )
            for perturbed_image in y])

        # Choose upper thresholds in binary searchs based on norm.
        if self.norm == 'Linf':
            highs = dists_post_update.to(self.device)
            thresholds = torch.clip(dists_post_update * theta, max=theta).to(self.device)
        else:
            highs = torch.ones(len(y)).to(self.device)
            thresholds = theta
        
        lows = torch.zeros(len(y)).to(self.device)
        while torch.max((highs - lows) / thresholds) > 1.0:
            mids = (highs + lows) / 2.0
            mid_images = self.project(x, y, mids)
            decisions = self.decision_function(mid_images, labels)
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)

            out_images = self.project(x, y, highs)
            # Compute distance of the output image to select the best choice.
            # (only used when stepsize_search is grid_search.)
            dists = torch.tensor([self.compute_distance(
                x,
                out_image,
            )
                for out_image in out_images])
            idx = torch.argmin(dists)
            dist = dists_post_update[0]
            out_image = out_images[idx]

        return out_image, dist

    def geometric_progression_for_stepsize(self, x, update, dist, cur_iter, labels):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        
        epsilon = dist / torch.sqrt(cur_iter)
        def phi(epsilon):
            new = x + epsilon * update
            success = self.decision_function(new[None], labels)
            return success

        while not phi(epsilon):
            epsilon /= 2.0

        return epsilon

    def define_delta(self, x, dist_post_update, cur_iter):
        """
        Choose the delta at the scale of distance
        between x and y x.
        """
        theta  = self.define_theta(x)
        if cur_iter == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.norm == 'L2':
                delta = torch.sqrt(torch.tensor(x.numel())) * theta * dist_post_update
            elif self.norm == 'Linf':
                delta = x.numel() * theta * dist_post_update

        return delta

    def initialize(self, x, labels):
        """
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        
        success = 0
        num_evals = 0
        if self.target_image is None:
            # Find a misclassified random noise.
            while True:
                random_noise = ((self.clip_max - self.clip_min) * torch.rand(x.shape) + self.clip_min).to(self.device)
                success = self.decision_function(random_noise[None], labels)[0]
                num_evals += 1
                if int(success):
                    break
                assert num_evals < 1e4, "Initialization failed! "
                "Use a misclassified image as `target_image`"

            # Binary search to minimize L2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * x + mid * random_noise
                success = self.decision_function(blended[None], labels)
                if int(success):
                    high = mid
                else:
                    low = mid

            initialization = (1 - high) * x + high * random_noise

        else:
            initialization = self.target_image

        return initialization

    def perturb(self, x, labels):
            # Initialize.
        y = self.initialize(x, labels)

        # Project the initialization to the boundary.
        y, dist_post_update = self.binary_search_batch(x, torch.unsqueeze(y, 0).to(self.device), labels)
        dist = self.compute_distance(y, x)

        for j in tqdm(torch.arange(self.num_iterations)):
            # Choose delta.
            cur_iter = j + 1
            delta = self.define_delta(x, dist_post_update, cur_iter)

            # Choose number of evaluations.
            num_evals = int(self.init_num_evals * torch.sqrt(j + 1))
            num_evals = int(min([num_evals, self.max_num_evals]))

            # approximate gradient.
            gradf = self.approximate_gradient(y, num_evals,
                                              delta, labels)
            if self.norm == 'Linf':
                update = torch.sign(gradf)
            else:
                update = gradf
            update = update.type(torch.float).to(self.device)
            # search step size.
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(y,
                                                                  update, dist, cur_iter, labels)

                # Update the x.
                y = self.clip_image(y + epsilon * update,
                                            self.clip_min, self.clip_max)

                # Binary search to return to the boundary.
                y, dist_post_update = self.binary_search_batch(x, y[None],labels)

            elif self.stepsize_search == 'grid_search':
                # Grid search for stepsize.
                coff = torch.logspace(-4, 0, steps=20).to(self.device)
                epsilons = coff * dist
                epsilons_shape = [20] + len(x.shape) * [1]
                perturbeds = y + epsilons.reshape(epsilons_shape) * update
                perturbeds = self.clip_image(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = self.decision_function(perturbeds, labels)

                if torch.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    y, dist_post_update = self.binary_search_batch(x, perturbeds[idx_perturbed], labels)

            # compute new distance.
            dist = self.compute_distance(y, x)
            if self.verbose:
                print('iteration: {:d}, {:s} distance {:.4E}'.format(j + 1, self.norm, dist))

        return y
