import torch
import torch.nn as nn
import torch.optim as optim

class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str): name of an attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._transform_label = self._get_label
        self._targeted = -1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._kth_min = 1

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
            
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.
        """
        if self._attack_mode is 'only_default':
            self._attack_mode = "only_default"
        else:
            self._attack_mode = "default"
            
        self._targeted = -1
        self._transform_label = self._get_label
        
    def set_mode_targeted(self, target_map_function=None):
        r"""
        Set attack mode as targeted mode.
  
        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (DEFAULT)
        """
        if self._attack_mode is 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        self._attack_mode = "targeted"
        self._targeted = 1
        if target_map_function is None:
            self._target_map_function = lambda images, labels:labels
        else:
            self._target_map_function = target_map_function
        self._transform_label = self._get_target_label
        
        
    def set_mode_least_likely(self, kth_min=1):
        r"""
        Set attack mode as least likely mode.
  
        Arguments:
            kth_min (str): k-th smallest probability used as target labels (DEFAULT: 1)
        """
        if self._attack_mode is 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        self._attack_mode = "least_likely"
        self._targeted = 1
        self._transform_label = self._get_least_likely_label
        self._kth_min = kth_min
        
        
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str): 'float' or 'int'. (DEFAULT: 'float')
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def save(self, data_loader, save_path=None, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (DEFAULT: True)
        """
        if (self._attack_mode is 'targeted') and (self._target_map_function is None):
            raise ValueError("save is not supported for target_map_function=None")
        
        if save_path is not None:
            image_list = []
            label_list = []

        correct = 0
        total = 0
        l2_distance = []
        
        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            batch_size = len(images)
            
            if save_path is not None:
                image_list.append(adv_images.cpu())
                label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (predicted == labels.to(self.device))
                    correct += right_idx.sum()
                    
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))                    
                    acc = 100 * float(correct) / total
                    print('- Save Progress: %2.2f %% / Accuracy: %2.2f %% / L2: %1.5f' \
                          % ((step+1)/total_batch*100, acc, torch.cat(l2_distance).mean()), end='\r')

        if save_path is not None:
            x = torch.cat(image_list, 0)
            y = torch.cat(label_list, 0)            
            torch.save((x, y), save_path)
            print('\n- Save Complete!')

        self._switch_model()
        
    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels
    
    def _get_target_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return self._target_map_function(images, labels)
    
    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, labels = torch.kthvalue(outputs.data, pos)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default':
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    Distance Measure : L2
    
    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (DEFAULT: 1e-4)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`    
        kappa (float): kappa (also written as 'confidence') in the paper. (DEFAULT: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (DEFAULT: 1000)
        lr (float): learning rate of the Adam optimizer. (DEFAULT: 0.01)
        
    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)
        
    .. note:: NOT IMPLEMENTED methods in the paper due to time consuming.
    
        (1) Binary search for c.
    """
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super(CW, self).__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True
        
        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)
        
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)
            
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()
            
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2
            
            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
            
            # Early Stop when loss does not converge.
            if step % (self.steps//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()
                
        return best_adv_images
    
    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1) 
    
    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(self._targeted*(i-j), min=-self.kappa)