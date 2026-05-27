from action_provider.action_provider_dds import DDSActionProvider
from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wh_dds import DDSRLActionProvider
from action_provider.action_provider_bc import ActionProviderBC


def create_action_provider(env, args):
    if args.action_source == "dds":
        return DDSActionProvider(env=env, args_cli=args)
    elif args.action_source == "dds_wholebody":
        return DDSRLActionProvider(env=env, args_cli=args)
    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env, args_cli=args)
    elif args.action_source == "policy":
        from .action_provider_bc import ActionProviderBC
        return ActionProviderBC(env, args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None
