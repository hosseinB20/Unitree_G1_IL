from action_provider.action_provider_act import ActionProviderACT


def create_action_provider(env, args_cli):
    print(f"create action provider: {args_cli.action_source}")

    if args_cli.action_source.lower() == "act":
        return ActionProviderACT(env, args_cli)

    raise ValueError(f"Unknown action provider: {args_cli.action_source}")